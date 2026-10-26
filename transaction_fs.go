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
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/spf13/afero/mem"
)

var (
	_ Fs      = (*txnFs)(nil)
	_ Lstater = (*txnFs)(nil)
)

// txnFs is the scoped Fs view used while a WriteScope is active.
//
// Mutations are recorded in a private in-memory overlay plus an ordered
// journal; the underlying Fs is never modified before commit. Reads merge
// the overlay, an exact-path tombstone set and the underlying Fs.
type txnFs struct {
	// top is the Fs the scope was opened on (for a nested scope it is the
	// outer scoped view).
	top Fs

	scope  *WriteScope
	parent *txnFs

	state *int32 // alias of WriteScope view state

	overlay  *MemMapFs
	deleted  map[string]struct{}
	implicit map[*mem.FileData]struct{}
	journal  []txOp

	// Transparent wrapper chain collected for rule replication.
	readOnly  bool
	basepaths []*BasePathFs // outer -> inner
	regexps   []*RegexpFs   // outer -> inner
}

func newTxnFs(base Fs, scope *WriteScope, parent *WriteScope) *txnFs {
	state := int32(scopeStateActive)
	t := &txnFs{
		top:      base,
		scope:    scope,
		overlay:  &MemMapFs{},
		deleted:  map[string]struct{}{},
		implicit: map[*mem.FileData]struct{}{},
		state:    &state,
	}
	if parent != nil {
		t.parent = parent.view
	}
	t.collectChain(base)
	return t
}

// collectChain walks the transparent wrapper layers (BasePathFs,
// RegexpFs, HttpFs, ReadOnlyFs) so the scoped view can enforce the exact
// same write/read rules while mutations are staged.
func (t *txnFs) collectChain(fs Fs) {
	cur := fs
	for {
		switch v := cur.(type) {
		case *ReadOnlyFs:
			t.readOnly = true
			return
		case *BasePathFs:
			t.basepaths = append(t.basepaths, v)
			cur = v.source
		case *RegexpFs:
			t.regexps = append(t.regexps, v)
			cur = v.source
		default:
			return
		}
	}
}

func (*txnFs) Name() string { return "WriteScopeFs" }

func (t *txnFs) active() bool {
	return t.state != nil && atomic.LoadInt32(t.state) == scopeStateActive
}

// ---------------------------------------------------------------------------
// Name/rule gates
// ---------------------------------------------------------------------------

// validateBasePath applies every BasePathFs rule in chain order. It returns
// the error the corresponding BasePathFs method would have returned.
func (t *txnFs) validateBasePath(name, op string) error {
	cur := name
	for _, bp := range t.basepaths {
		real, err := bp.RealPath(cur)
		if err != nil {
			return &os.PathError{Op: op, Path: cur, Err: err}
		}
		cur = real
	}
	return nil
}

func (t *txnFs) matchesName(name string) error {
	for _, r := range t.regexps {
		if r.re == nil {
			continue
		}
		if !r.re.MatchString(filepath.Base(name)) {
			return syscall.ENOENT
		}
	}
	return nil
}

// rawIsDir reports whether the name resolves to a directory in the scoped
// view, without running the public (gated) Stat again.
func (t *txnFs) rawIsDir(name string) (bool, error) {
	fi, _, err := t.statRaw(name)
	if err != nil {
		return false, err
	}
	return fi.IsDir(), nil
}

// dirOrMatches mirrors RegexpFs.dirOrMatches but resolves existence from
// the scoped view so staged entries are visible to the rule.
func (t *txnFs) dirOrMatches(name string) error {
	dir, err := t.rawIsDir(name)
	if err != nil {
		return err
	}
	if dir {
		return nil
	}
	return t.matchesName(name)
}

// ---------------------------------------------------------------------------
// Overlay primitives
// ---------------------------------------------------------------------------

func (t *txnFs) overlayGet(key string) (*mem.FileData, bool) {
	t.overlay.mu.RLock()
	fd, ok := t.overlay.getData()[key]
	t.overlay.mu.RUnlock()
	return fd, ok
}

func (t *txnFs) overlayChildren(parentKey string) []*mem.FileData {
	t.overlay.mu.RLock()
	var out []*mem.FileData
	for p, fd := range t.overlay.getData() {
		if p == parentKey {
			continue
		}
		if normalizePath(filepath.Dir(p)) == parentKey {
			out = append(out, fd)
		}
	}
	t.overlay.mu.RUnlock()
	return out
}

// markImplicitAncestors flags the auto-created directory parents of key as
// implicit (they should delegate to the underlying Fs whenever it has the
// same directory).
func (t *txnFs) markImplicitAncestors(key string) {
	dir := normalizePath(filepath.Dir(key))
	for dir != key && dir != FilePathSeparator {
		fd, ok := t.overlayGet(dir)
		if !ok {
			return
		}
		t.implicit[fd] = struct{}{}
		next := normalizePath(filepath.Dir(dir))
		if next == dir {
			return
		}
		dir = next
	}
}

// sweepImplicit removes implicit, empty, underlying-less directory
// ancestors left behind by removes/renames so the view does not expose
// phantom staging directories.
func (t *txnFs) sweepImplicit(startKey string) {
	dir := normalizePath(filepath.Dir(startKey))
	for dir != FilePathSeparator && dir != "." && dir != "" {
		fd, ok := t.overlayGet(dir)
		if !ok {
			return
		}
		if _, implicit := t.implicit[fd]; !implicit {
			return
		}
		if len(t.overlayChildren(dir)) > 0 {
			return
		}
		if _, err := t.top.Stat(dir); err == nil {
			return
		}
		_ = t.overlay.Remove(dir)
		delete(t.implicit, fd)
		next := normalizePath(filepath.Dir(dir))
		if next == dir {
			return
		}
		dir = next
	}
}

// injectFile creates a private regular-file entry in the overlay.
func (t *txnFs) injectFile(key string, content []byte, fi os.FileInfo) *mem.FileData {
	h, err := t.overlay.Create(key)
	if err != nil {
		panic(err) // overlay Create cannot fail
	}
	mf := h.(*mem.File)
	fd := mf.Data()
	if len(content) > 0 {
		_, _ = mf.Write(content)
	}
	if fi != nil {
		mem.SetMode(fd, fi.Mode())
		mem.SetModTime(fd, fi.ModTime())
	}
	return fd
}

// injectDir creates a private directory entry in the overlay, cloning the
// metadata of fi when given.
func (t *txnFs) injectDir(key string, fi os.FileInfo) *mem.FileData {
	if err := t.overlay.lockfreeMkdir(key, 0o755); err != nil {
		// Already present (directory): reuse it.
		if fd, ok := t.overlayGet(key); ok {
			return fd
		}
		panic(err)
	}
	fd, _ := t.overlay.lockfreeOpen(key)
	if fi != nil {
		mem.SetMode(fd, fi.Mode())
		mem.SetModTime(fd, fi.ModTime())
	}
	return fd
}

// snapshotFile copies an underlying file into the private overlay without
// ever holding a writable handle to the underlying file.
func (t *txnFs) snapshotFile(name, key string, truncate bool) *mem.FileData {
	fi, err := t.top.Stat(name)
	if err != nil {
		return nil
	}
	var content []byte
	if !truncate && !fi.IsDir() {
		content, _ = ReadFile(t.top, name)
	}
	return t.injectFile(key, content, fi)
}

// ----------------------------------------------------------------------------
// View resolution
// ----------------------------------------------------------------------------

// statRaw resolves a name against overlay -> tombstone -> underlying Fs.
// The bool reports whether the result lives in the overlay.
func (t *txnFs) statRaw(name string) (os.FileInfo, bool, error) {
	key := normalizePath(name)
	if fd, ok := t.overlayGet(key); ok {
		if _, implicit := t.implicit[fd]; implicit {
			if fi, err := t.top.Stat(name); err == nil {
				return fi, false, nil
			}
		}
		return mem.GetFileInfo(fd), true, nil
	}
	if _, del := t.deleted[key]; del {
		return nil, false, &os.PathError{Op: "stat", Path: name, Err: os.ErrNotExist}
	}
	fi, err := t.top.Stat(name)
	return fi, false, err
}

func notExist(op, name string) error {
	return &os.PathError{Op: op, Path: name, Err: os.ErrNotExist}
}

func existError(op, name string) error {
	return &os.PathError{Op: op, Path: name, Err: ErrFileExists}
}

// ----------------------------------------------------------------------------
// Fs: read path
// ----------------------------------------------------------------------------

func (t *txnFs) Stat(name string) (os.FileInfo, error) {
	if err := t.validateBasePath(name, "stat"); err != nil {
		return nil, err
	}
	if err := t.dirOrMatches(name); err != nil {
		return nil, err
	}
	fi, _, err := t.statRaw(name)
	return fi, err
}

func (t *txnFs) LstatIfPossible(name string) (os.FileInfo, bool, error) {
	if err := t.validateBasePath(name, "lstat"); err != nil {
		return nil, false, err
	}
	if err := t.dirOrMatches(name); err != nil {
		return nil, false, err
	}
	key := normalizePath(name)
	if fd, ok := t.overlayGet(key); ok {
		if _, implicit := t.implicit[fd]; implicit {
			if fi, err := t.top.Stat(name); err == nil {
				if lf, ok := t.top.(Lstater); ok {
					return lf.LstatIfPossible(name)
				}
				return fi, false, nil
			}
		}
		return mem.GetFileInfo(fd), false, nil
	}
	if _, del := t.deleted[key]; del {
		return nil, false, notExist("lstat", name)
	}
	if lf, ok := t.top.(Lstater); ok {
		return lf.LstatIfPossible(name)
	}
	fi, err := t.top.Stat(name)
	return fi, false, err
}

func (t *txnFs) Open(name string) (File, error) {
	if err := t.validateBasePath(name, "open"); err != nil {
		return nil, err
	}
	// Replicate RegexpFs.Open.
	if dir, err := t.rawIsDir(name); err != nil {
		return nil, err
	} else if !dir {
		if err := t.matchesName(name); err != nil {
			return nil, err
		}
	}
	return t.openRead(name, os.O_RDONLY, 0)
}

func (t *txnFs) OpenFile(name string, flag int, perm os.FileMode) (File, error) {
	if t.readOnly && flag&(os.O_WRONLY|syscall.O_RDWR|os.O_APPEND|os.O_CREATE|os.O_TRUNC) != 0 {
		return nil, syscall.EPERM
	}
	if err := t.validateBasePath(name, "openfile"); err != nil {
		return nil, err
	}
	// Replicate RegexpFs.OpenFile.
	if err := t.dirOrMatches(name); err != nil {
		if flag&os.O_CREATE == 0 || !os.IsNotExist(err) {
			return nil, err
		}
		if err := t.matchesName(name); err != nil {
			return nil, err
		}
	}

	write := flag&(os.O_WRONLY|syscall.O_RDWR|os.O_APPEND|os.O_CREATE|os.O_TRUNC) != 0
	if !write {
		return t.openRead(name, flag, perm)
	}
	return t.openWrite(name, flag, perm)
}

func (t *txnFs) openRead(name string, flag int, perm os.FileMode) (File, error) {
	key := normalizePath(name)
	if fd, ok := t.overlayGet(key); ok {
		if mem.GetFileInfo(fd).IsDir() {
			return t.newDirHandle(name, key, fd), nil
		}
		// Match MemMapFs.Open read-permission check.
		if mem.GetFileInfo(fd).Mode().Perm()&0o444 == 0 {
			return nil, &os.PathError{Op: "open", Path: name, Err: os.ErrPermission}
		}
		return mem.NewReadOnlyFileHandle(fd), nil
	}
	if _, del := t.deleted[key]; del {
		return nil, notExist("open", name)
	}
	fi, err := t.top.Stat(name)
	if err == nil && fi.IsDir() {
		return t.newDirHandle(name, key, nil), nil
	}
	// Plain read of an underlying file: delegate to a read-only underlying
	// handle, which can never write through.
	return t.top.OpenFile(name, flag, perm)
}

func (t *txnFs) openWrite(name string, flag int, perm os.FileMode) (File, error) {
	key := normalizePath(name)

	if fd, ok := t.overlayGet(key); ok {
		if flag&os.O_EXCL != 0 {
			return nil, existError("open", name)
		}
		h, err := t.overlay.OpenFile(key, flag, perm)
		if err != nil {
			return nil, err
		}
		lazy := flag&os.O_TRUNC == 0
		return t.newRWFile(h.(*mem.File), fd, lazy), nil
	}

	topFi, topErr := t.top.Stat(name)
	_, isDeleted := t.deleted[key]
	exists := topErr == nil && !isDeleted
	if exists {
		if flag&os.O_EXCL != 0 {
			return nil, existError("open", name)
		}
		// Match MemMapFs.openWrite write-permission check.
		if !topFi.IsDir() && topFi.Mode().Perm()&0o222 == 0 {
			return nil, &os.PathError{Op: "open", Path: name, Err: os.ErrPermission}
		}
		// Copy-up into a private staging entry. The underlying file is only
		// read, never opened for writing.
		fi := topFi
		var fd *mem.FileData
		if fi.IsDir() {
			fd = t.injectDir(key, fi)
		} else {
			fd = t.snapshotFile(name, key, flag&os.O_TRUNC != 0)
		}
		h := mem.NewFileHandle(fd)
		if flag&os.O_APPEND != 0 {
			if _, err := h.Seek(0, io.SeekEnd); err != nil {
				return nil, err
			}
		}
		lazy := true
		if flag&os.O_TRUNC != 0 && !fi.IsDir() {
			// Truncation itself is a staged mutation: journal it now.
			t.journal = append(t.journal, txOp{kind: opWrite, path: name, fd: fd})
			lazy = false
		}
		return t.newRWFile(h, fd, lazy), nil
	}

	if flag&os.O_CREATE == 0 {
		if os.IsNotExist(topErr) {
			return nil, notExist("open", name)
		}
		return nil, topErr
	}

	// Brand new file in the overlay.
	h, err := t.overlay.Create(key)
	if err != nil {
		return nil, err
	}
	mf := h.(*mem.File)
	fd := mf.Data()
	mem.SetMode(fd, perm&chmodBits)
	t.markImplicitAncestors(key)
	t.journal = append(t.journal, txOp{kind: opWrite, path: name, fd: fd})
	return t.newRWFile(mf, fd, false), nil
}

// ----------------------------------------------------------------------------
// Fs: write path
// ----------------------------------------------------------------------------

func (t *txnFs) Create(name string) (File, error) {
	if t.readOnly {
		return nil, syscall.EPERM
	}
	if err := t.matchesName(name); err != nil {
		return nil, err
	}
	if err := t.validateBasePath(name, "create"); err != nil {
		return nil, err
	}
	key := normalizePath(name)
	h, err := t.overlay.Create(key)
	if err != nil {
		return nil, err
	}
	mf := h.(*mem.File)
	fd := mf.Data()
	mem.SetMode(fd, 0o666)
	t.markImplicitAncestors(key)
	t.journal = append(t.journal, txOp{kind: opWrite, path: name, fd: fd})
	return t.newRWFile(mf, fd, false), nil
}

func (t *txnFs) Mkdir(name string, perm os.FileMode) error {
	if t.readOnly {
		return syscall.EPERM
	}
	if err := t.validateBasePath(name, "mkdir"); err != nil {
		return err
	}
	key := normalizePath(name)
	if _, _, err := t.statRaw(name); err == nil {
		return existError("mkdir", name)
	}
	perm &= chmodBits
	if err := t.overlay.Mkdir(key, perm); err != nil {
		return err
	}
	t.markImplicitAncestors(key)
	t.journal = append(t.journal, txOp{kind: opMkdir, path: name, perm: perm})
	return nil
}

func (t *txnFs) MkdirAll(name string, perm os.FileMode) error {
	if t.readOnly {
		if fi, err := t.top.Stat(name); err == nil && fi.IsDir() {
			return nil
		}
		return syscall.EPERM
	}
	if err := t.validateBasePath(name, "mkdir"); err != nil {
		return err
	}
	_, _, statErr := t.statRaw(name)
	if statErr == nil {
		// MemMapFs.MkdirAll returns nil when the path already exists,
		// whether it is a directory or a file.
		return nil
	}
	key := normalizePath(name)
	perm &= chmodBits
	if err := t.overlay.MkdirAll(key, perm); err != nil {
		return err
	}
	t.markImplicitAncestors(key)
	t.journal = append(t.journal, txOp{kind: opMkdirAll, path: name, perm: perm})
	return nil
}

func (t *txnFs) Remove(name string) error {
	if t.readOnly {
		return syscall.EPERM
	}
	if err := t.dirOrMatches(name); err != nil {
		return err
	}
	if err := t.validateBasePath(name, "remove"); err != nil {
		return err
	}
	key := normalizePath(name)
	fi, _, err := t.statRaw(name)
	if err != nil {
		return &os.PathError{Op: "remove", Path: name, Err: os.ErrNotExist}
	}
	// A directory with visible children cannot be removed, mirroring
	// os.Remove; use RemoveAll for whole trees.
	if fi.IsDir() {
		count := 0
		_ = Walk(t, name, func(p string, info os.FileInfo, err error) error {
			if err == nil {
				count++
			}
			return nil
		})
		if count > 1 {
			return &os.PathError{Op: "remove", Path: name, Err: syscall.ENOTEMPTY}
		}
	}
	if _, inOverlay := t.overlayGet(key); inOverlay {
		if err := t.overlay.Remove(key); err != nil {
			return err
		}
	}
	// Tombstone whenever the underlying Fs currently has the path, so a
	// staged delete hides a pre-existing entry until commit.
	if _, err := t.top.Stat(name); err == nil {
		t.deleted[key] = struct{}{}
	}
	t.sweepImplicit(key)
	t.journal = append(t.journal, txOp{kind: opRemove, path: name})
	return nil
}

func (t *txnFs) RemoveAll(name string) error {
	if t.readOnly {
		return syscall.EPERM
	}
	// Replicate RegexpFs.RemoveAll.
	if dir, err := t.rawIsDir(name); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	} else {
		if !dir {
			if err := t.matchesName(name); err != nil {
				return err
			}
		}
	}
	if err := t.validateBasePath(name, "remove_all"); err != nil {
		return err
	}

	key := normalizePath(name)
	_, _, statErr := t.statRaw(name)
	underlyingHas := false
	if _, err := t.top.Stat(name); err == nil {
		underlyingHas = true
	}

	// Snapshot the visible subtree and tombstone every path that exists
	// (now) in the underlying Fs.
	if statErr == nil {
		var paths []string
		_ = Walk(t, name, func(p string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			paths = append(paths, p)
			return nil
		})
		for _, p := range paths {
			if _, err := t.top.Stat(p); err == nil {
				t.deleted[normalizePath(p)] = struct{}{}
			}
		}
	}

	_ = t.overlay.RemoveAll(key)
	t.sweepImplicit(key)

	if statErr == nil || underlyingHas {
		t.journal = append(t.journal, txOp{kind: opRemoveAll, path: name})
	}
	return nil
}

func (t *txnFs) Rename(oldname, newname string) error {
	if t.readOnly {
		return syscall.EPERM
	}
	// Replicate RegexpFs.Rename: when a RegexpFs layer is present, renaming
	// a directory is a no-op success.
	if len(t.regexps) > 0 {
		if dir, err := t.rawIsDir(oldname); err == nil && dir {
			return nil
		}
	}
	if err := t.matchesName(oldname); err != nil {
		return err
	}
	if err := t.matchesName(newname); err != nil {
		return err
	}
	if err := t.validateBasePath(oldname, "rename"); err != nil {
		return err
	}
	if err := t.validateBasePath(newname, "rename"); err != nil {
		return err
	}

	oldKey := normalizePath(oldname)
	newKey := normalizePath(newname)
	if oldKey == newKey {
		return nil
	}
	if newKey != FilePathSeparator &&
		(newKey == oldKey || strings.HasPrefix(newKey, oldKey+FilePathSeparator)) {
		return &os.LinkError{Op: "rename", Old: oldname, New: newname, Err: syscall.EINVAL}
	}

	if _, _, err := t.statRaw(oldname); err != nil {
		return &os.PathError{Op: "rename", Path: oldname, Err: os.ErrNotExist}
	}

	// Enumerate the visible subtree rooted at the old name.
	var subtree []string
	if err := Walk(t, oldname, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		subtree = append(subtree, p)
		return nil
	}); err != nil {
		return err
	}
	sort.Slice(subtree, func(i, j int) bool {
		return strings.Count(subtree[i], FilePathSeparator) <
			strings.Count(subtree[j], FilePathSeparator)
	})

	_, oldInOverlay := t.overlayGet(oldKey)

	// Clone underlying-only entries at their new names first.
	for _, p := range subtree {
		pKey := normalizePath(p)
		if _, inOverlay := t.overlayGet(pKey); inOverlay {
			continue
		}
		rel := ""
		if pKey != oldKey {
			rel = strings.TrimPrefix(pKey, oldKey)
		}
		nKey := newKey + rel
		fi, err := t.top.Stat(p)
		if err != nil {
			continue
		}
		if fi.IsDir() {
			t.injectDir(nKey, fi)
		} else {
			t.snapshotFile(p, nKey, false)
		}
		// Parents created for the cloned subtree are implicit.
		t.markImplicitAncestors(nKey)
	}

	// Tombstone the whole visible old subtree.
	for _, p := range subtree {
		t.deleted[normalizePath(p)] = struct{}{}
	}

	// Move the overlay subtree (if its root was staged) onto the clones.
	if oldInOverlay {
		if err := t.overlay.Rename(oldKey, newKey); err != nil {
			return err
		}
	}

	t.sweepImplicit(oldKey)
	t.journal = append(t.journal, txOp{kind: opRename, path: oldname, path2: newname})
	return nil
}

func (t *txnFs) Chmod(name string, mode os.FileMode) error {
	if t.readOnly {
		return syscall.EPERM
	}
	if err := t.dirOrMatches(name); err != nil {
		return err
	}
	if err := t.validateBasePath(name, "chmod"); err != nil {
		return err
	}
	key := normalizePath(name)
	fi, inOverlay, err := t.statRaw(name)
	if err != nil {
		return &os.PathError{Op: "chmod", Path: name, Err: os.ErrNotExist}
	}
	if !inOverlay {
		if fi.IsDir() {
			t.injectDir(key, fi)
		} else {
			t.snapshotFile(name, key, false)
		}
	}
	if err := t.overlay.Chmod(key, mode); err != nil {
		return err
	}
	t.journal = append(t.journal, txOp{kind: opChmod, path: name, perm: mode & chmodBits})
	return nil
}

func (t *txnFs) Chown(name string, uid, gid int) error {
	if t.readOnly {
		return syscall.EPERM
	}
	if err := t.dirOrMatches(name); err != nil {
		return err
	}
	if err := t.validateBasePath(name, "chown"); err != nil {
		return err
	}
	key := normalizePath(name)
	fi, inOverlay, err := t.statRaw(name)
	if err != nil {
		return &os.PathError{Op: "chown", Path: name, Err: os.ErrNotExist}
	}
	if !inOverlay {
		if fi.IsDir() {
			t.injectDir(key, fi)
		} else {
			t.snapshotFile(name, key, false)
		}
	}
	if err := t.overlay.Chown(key, uid, gid); err != nil {
		return err
	}
	t.journal = append(t.journal, txOp{kind: opChown, path: name, uid: uid, gid: gid})
	return nil
}

func (t *txnFs) Chtimes(name string, atime, mtime time.Time) error {
	if t.readOnly {
		return syscall.EPERM
	}
	if err := t.dirOrMatches(name); err != nil {
		return err
	}
	if err := t.validateBasePath(name, "chtimes"); err != nil {
		return err
	}
	key := normalizePath(name)
	fi, inOverlay, err := t.statRaw(name)
	if err != nil {
		return &os.PathError{Op: "chtimes", Path: name, Err: os.ErrNotExist}
	}
	if !inOverlay {
		if fi.IsDir() {
			t.injectDir(key, fi)
		} else {
			t.snapshotFile(name, key, false)
		}
	}
	if err := t.overlay.Chtimes(key, atime, mtime); err != nil {
		return err
	}
	t.journal = append(t.journal, txOp{kind: opChtimes, path: name, atime: atime, mtime: mtime})
	return nil
}

// ----------------------------------------------------------------------------
// Nesting, rollback, commit
// ----------------------------------------------------------------------------

// mergeChild applies an inner scope's staging to this outer view. Called
// when an inner scope commits. Nothing reaches the underlying Fs.
func (t *txnFs) mergeChild(child *txnFs) {
	// Apply the child's tombstones to the parent overlay.
	for p := range child.deleted {
		if _, ok := t.overlayGet(p); ok {
			_ = t.overlay.Remove(p)
		}
		t.deleted[p] = struct{}{}
	}

	// Merge the child overlay entries, shallowest first so parenting is
	// intact. Child entries shadow parent entries with the same name.
	type kv struct {
		key string
		fd  *mem.FileData
	}
	var entries []kv
	child.overlay.mu.RLock()
	for p, fd := range child.overlay.getData() {
		if p == FilePathSeparator {
			continue
		}
		entries = append(entries, kv{p, fd})
	}
	child.overlay.mu.RUnlock()
	sort.Slice(entries, func(i, j int) bool {
		return strings.Count(entries[i].key, FilePathSeparator) <
			strings.Count(entries[j].key, FilePathSeparator)
	})

	t.overlay.mu.Lock()
	data := t.overlay.getData()
	for _, e := range entries {
		if old, ok := data[e.key]; ok && old != e.fd {
			t.overlay.unRegisterWithParent(e.key)
		}
		data[e.key] = e.fd
		t.overlay.registerWithParent(e.fd, 0)
	}
	t.overlay.mu.Unlock()

	for fd := range child.implicit {
		t.implicit[fd] = struct{}{}
	}

	// The child's journal extends the parent's in occurrence order.
	t.journal = append(t.journal, child.journal...)
	child.discard()
}

// discard drops all staging references. The underlying Fs has never been
// mutated by the scope itself, so dropping memory is a complete rollback.
func (t *txnFs) discard() {
	t.overlay = nil
	t.deleted = nil
	t.implicit = nil
	t.journal = nil
}

// commitToBase replays the journal onto the underlying Fs. When the
// underlying Fs is a MemMapFs (optionally behind BasePathFs layers), the
// whole journal is applied under a single write lock, giving other readers
// an all-or-nothing visibility switch. Any other Fs gets a strictly
// ordered replay through its regular write methods.
func (t *txnFs) commitToBase() error {
	commitMuFor(t.top).Lock()
	defer commitMuFor(t.top).Unlock()

	if m, translate, ok := resolveMemTarget(t.top); ok {
		ops := make([]txOp, len(t.journal))
		copy(ops, t.journal)
		for i := range ops {
			rp, err := translate(ops[i].path)
			if err != nil {
				return err
			}
			ops[i].path = rp
			if ops[i].path2 != "" {
				rp, err := translate(ops[i].path2)
				if err != nil {
					return err
				}
				ops[i].path2 = rp
			}
		}
		return m.txApplyJournal(ops)
	}

	return t.replayGeneric()
}

func (t *txnFs) replayGeneric() error {
	for _, op := range t.journal {
		switch op.kind {
		case opWrite:
			dir := filepath.Dir(op.path)
			if dir != "" && dir != "." && dir != FilePathSeparator {
				if err := t.top.MkdirAll(dir, 0o755); err != nil {
					return err
				}
			}
			mode := os.FileMode(0o666)
			if op.fd != nil {
				mode = mem.GetFileInfo(op.fd).Mode().Perm()
			}
			f, err := t.top.OpenFile(op.path, os.O_CREATE|os.O_RDWR|os.O_TRUNC, mode)
			if err != nil {
				return err
			}
			if op.fd != nil {
				rh := mem.NewReadOnlyFileHandle(op.fd)
				n, err := io.Copy(f, rh)
				_ = rh.Close()
				if err != nil {
					_ = f.Close()
					return err
				}
				if n != mem.GetFileInfo(op.fd).Size() {
					_ = f.Close()
					return io.ErrShortWrite
				}
			}
			if err := f.Close(); err != nil {
				return err
			}
		case opMkdir:
			if dir := filepath.Dir(op.path); dir != "" && dir != "." && dir != FilePathSeparator {
				if err := t.top.MkdirAll(dir, 0o755); err != nil {
					return err
				}
			}
			if err := t.top.Mkdir(op.path, op.perm); err != nil {
				return err
			}
		case opMkdirAll:
			if err := t.top.MkdirAll(op.path, op.perm); err != nil {
				return err
			}
		case opRemove:
			if err := t.top.Remove(op.path); err != nil {
				return err
			}
		case opRemoveAll:
			if err := t.top.RemoveAll(op.path); err != nil {
				return err
			}
		case opRename:
			if dir := filepath.Dir(op.path2); dir != "" && dir != "." && dir != FilePathSeparator {
				if err := t.top.MkdirAll(dir, 0o755); err != nil {
					return err
				}
			}
			if err := t.top.Rename(op.path, op.path2); err != nil {
				return err
			}
		case opChmod:
			if err := t.top.Chmod(op.path, op.perm); err != nil {
				return err
			}
		case opChtimes:
			if err := t.top.Chtimes(op.path, op.atime, op.mtime); err != nil {
				return err
			}
		case opChown:
			if err := t.top.Chown(op.path, op.uid, op.gid); err != nil {
				return err
			}
		}
	}
	return nil
}
