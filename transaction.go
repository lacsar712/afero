// Copyright © 2026 Steve Francia <spf@spf13.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package afero

import (
	"errors"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/spf13/afero/mem"
)

// Scope lifecycle states.
const (
	scopeStateActive     int32 = 0
	scopeStateCommitted  int32 = 1
	scopeStateRolledBack int32 = 2
)

// Errors returned by write scopes.
var (
	// ErrScopeBusy is returned when a scope object is entered while it is
	// already active in another (concurrent) task.
	ErrScopeBusy = errors.New("afero: write scope is already in use by another task")
	// ErrScopeReentry is returned when the task currently owning an active
	// scope tries to enter the same scope again before exiting it. Inner
	// scopes must be opened with BeginWriteScope on the scoped Fs instead.
	ErrScopeReentry = errors.New("afero: write scope cannot be re-entered by its owner before it exits")
	// ErrScopeInactive is returned when Commit is called on a scope that has
	// already ended. Rollback on an ended scope is a harmless no-op.
	ErrScopeInactive = errors.New("afero: write scope is not active")
	// ErrScopeOwner is returned when a scope is committed or rolled back
	// from a task other than the one that entered it.
	ErrScopeOwner = errors.New("afero: write scope is bound to another task")
)

// WriteScope is an explicitly opened write scope (a transaction) on an Fs.
//
// Mutations performed on the Fs returned by Enter/View are staged in a
// private overlay: they are visible to the owning task through that Fs but
// never reach the underlying Fs while the scope is active. Commit replays
// the staged changes onto the underlying Fs, in the order they happened,
// in one visibility switch. Rollback discards the staging area and leaves
// the underlying Fs as it was when the scope was entered.
//
// A scope object may only be entered by one task at a time, and the owning
// task cannot enter it again while it is active. Open an inner scope with
// BeginWriteScope on the active scoped Fs to get nested behavior.
//
// The idiomatic usage is:
//
//	s, err := afero.BeginWriteScope(fs)
//	if err != nil { return err }
//	scoped := s.View()
//	defer func() { _ = s.Rollback() }()
//	// ... mutate scoped ...
//	if err := work(); err != nil { return err } // deferred Rollback drops everything
//	if err := s.Commit(); err != nil { return err }
type WriteScope struct {
	// base is the Fs the scope was opened on.
	base Fs

	mu    sync.Mutex
	owner int64
	view  *txnFs
	// parent is set when this scope was opened inside another active scope.
	parent *WriteScope
	// children counts active inner scopes merged-opened on this scope's view.
	children int
}

var errInnerScopeActive = errors.New("afero: an inner write scope is still active")

// NewWriteScope creates an idle, unbound write scope for fs. Call Enter to
// bind the scope to the current task and obtain the scoped Fs.
func NewWriteScope(fs Fs) *WriteScope {
	return &WriteScope{base: fs}
}

// BeginWriteScope creates a write scope for fs and enters it immediately.
// The returned scope is active; use View to perform operations.
//
// If fs is the scoped Fs of an active outer scope, an inner scope is
// opened. Committing the inner scope merges its staging into the outer
// scope without touching the underlying Fs.
func BeginWriteScope(fs Fs) (*WriteScope, error) {
	s := NewWriteScope(fs)
	if _, err := s.Enter(); err != nil {
		return s, err
	}
	return s, nil
}

// Enter binds the scope to the calling task and returns the scoped Fs.
//
// It fails with ErrScopeBusy when the scope is active in another task and
// with ErrScopeReentry when the calling task already owns the scope. Once
// a scope has ended (Commit or Rollback) it may be entered again.
func (s *WriteScope) Enter() (Fs, error) {
	goid := currentGoroutineID()

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.view != nil && atomic.LoadInt32(s.view.state) == scopeStateActive {
		if s.owner == goid {
			return nil, ErrScopeReentry
		}
		return nil, ErrScopeBusy
	}

	var parent *WriteScope
	base := s.base
	if parentView, ok := base.(*txnFs); ok {
		parentScope := parentView.scope
		if atomic.LoadInt32(parentScope.view.state) != scopeStateActive {
			// The outer scope already ended: treat the scoped Fs as an
			// ordinary Fs rooted at its underlying Fs.
			base = parentScope.view.top
		} else if parentScope.owner != goid {
			return nil, ErrScopeBusy
		} else {
			parent = parentScope
			base = parentView
		}
	}

	s.owner = goid
	s.parent = parent
	if parent != nil {
		parent.mu.Lock()
		parent.children++
		parent.mu.Unlock()
	}
	s.view = newTxnFs(base, s, parent)
	return s.view, nil
}

// View returns the scoped Fs. It is valid while the scope is active and
// becomes inert after the scope ends.
func (s *WriteScope) View() Fs {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.view == nil {
		return nil
	}
	return s.view
}

// Commit ends the scope successfully.
//
// For an inner scope, staged changes are merged into the outer scope's
// staging; the underlying Fs is not touched. For an outermost scope, all
// staged changes are applied to the underlying Fs through its write path
// in the order they occurred, with a single visibility switch.
//
// On error the scope stays active so the caller can retry or roll back.
func (s *WriteScope) Commit() error {
	s.mu.Lock()
	view := s.view
	owner := s.owner
	s.mu.Unlock()

	if view == nil || atomic.LoadInt32(view.state) != scopeStateActive {
		return ErrScopeInactive
	}
	if owner != currentGoroutineID() {
		return ErrScopeOwner
	}
	if s.children > 0 {
		return errInnerScopeActive
	}

	if s.parent != nil {
		s.parent.view.mergeChild(view)
		atomic.StoreInt32(view.state, scopeStateCommitted)
		s.leaveParent()
		s.release()
		return nil
	}

	if err := view.commitToBase(); err != nil {
		return err
	}
	atomic.StoreInt32(view.state, scopeStateCommitted)
	s.release()
	return nil
}

// Rollback discards all staged changes and ends the scope. The underlying
// Fs is never modified by a rolled back scope. Rolling back an already
// ended scope is a no-op and returns nil, which makes
// "defer scope.Rollback()" safe next to Commit.
func (s *WriteScope) Rollback() error {
	s.mu.Lock()
	view := s.view
	owner := s.owner
	active := view != nil && atomic.LoadInt32(view.state) == scopeStateActive
	s.mu.Unlock()

	if !active {
		return nil
	}
	if owner != currentGoroutineID() {
		return ErrScopeOwner
	}
	if s.children > 0 {
		return errInnerScopeActive
	}
	view.discard()
	atomic.StoreInt32(view.state, scopeStateRolledBack)
	s.leaveParent()
	s.release()
	return nil
}

// leaveParent detaches this scope from an outer scope.
func (s *WriteScope) leaveParent() {
	if s.parent == nil {
		return
	}
	s.parent.mu.Lock()
	s.parent.children--
	s.parent.mu.Unlock()
}

// release clears the task binding. The scope object can be entered again
// afterwards.
func (s *WriteScope) release() {
	s.mu.Lock()
	s.owner = 0
	s.parent = nil
	s.mu.Unlock()
}

// txOpKind enumerates the journaled write-path operations.
type txOpKind int

const (
	opWrite txOpKind = iota
	opMkdir
	opMkdirAll
	opRemove
	opRemoveAll
	opRename
	opChmod
	opChtimes
	opChown
)

// txOp is one entry of a scope's ordered journal.
type txOp struct {
	kind  txOpKind
	path  string
	path2 string        // rename destination
	fd    *mem.FileData // opWrite content/mode source
	perm  os.FileMode   // mkdir/chmod modes
	atime time.Time     // chtimes
	mtime time.Time     // chtimes
	uid   int           // chown
	gid   int           // chown
}

// currentGoroutineID returns the id of the current goroutine. The id is
// only used to distinguish the scope owner from other tasks; goroutine
// ids are unique for the lifetime of a goroutine.
func currentGoroutineID() int64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	// The first line looks like "goroutine 18 [running, ...]:"
	line := buf[:n]
	if len(line) > 10 && string(line[:10]) == "goroutine " {
		rest := line[10:]
		end := strings.IndexByte(string(rest), ' ')
		if end > 0 {
			if id, err := strconv.ParseInt(string(rest[:end]), 10, 64); err == nil {
				return id
			}
		}
	}
	return 0
}

// commitMuFor returns the mutex serializing commits to a given top-level Fs.
var txCommitMutexes sync.Map // Fs -> *sync.Mutex

func commitMuFor(fs Fs) *sync.Mutex {
	if m, ok := txCommitMutexes.Load(fs); ok {
		return m.(*sync.Mutex)
	}
	m, _ := txCommitMutexes.LoadOrStore(fs, &sync.Mutex{})
	return m.(*sync.Mutex)
}
