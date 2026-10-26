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
	"os"
	"path/filepath"

	"github.com/spf13/afero/mem"
)

// resolveMemTarget reports whether top is a MemMapFs, possibly reached
// through transparent BasePathFs layers. The translate function converts
// scoped (logical) names into the names the target MemMapFs expects.
func resolveMemTarget(top Fs) (*MemMapFs, func(string) (string, error), bool) {
	var bps []*BasePathFs
	cur := top
	for {
		switch v := cur.(type) {
		case *MemMapFs:
			translate := func(name string) (string, error) {
				out := name
				for _, bp := range bps { // outer -> inner
					rn, err := bp.RealPath(out)
					if err != nil {
						return "", err
					}
					out = rn
				}
				return out, nil
			}
			return v, translate, true
		case *BasePathFs:
			bps = append(bps, v)
			cur = v.source
		default:
			return nil, nil, false
		}
	}
}

// txApplyJournal replays a scope journal on a MemMapFs. The whole replay
// happens under the Fs's write lock, which is the same lock all normal
// MemMapFs reads/writes already take: readers outside the scope see either
// the pre-commit or the post-commit tree, never an intermediate one.
//
// The locking rules of MemMapFs are unchanged; when no scope is active this
// code path is never entered.
func (m *MemMapFs) txApplyJournal(ops []txOp) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	data := m.getData()

	for _, op := range ops {
		switch op.kind {
		case opWrite:
			name := normalizePath(op.path)
			// A rename can have carried this FileData to another key; drop
			// that binding before installing it here.
			if prev := m.findKeyForLocked(data, op.fd); prev != "" && prev != name {
				m.unregisterLocked(data, prev)
				delete(data, prev)
			}
			if old := data[name]; old != nil && old != op.fd {
				m.unregisterLocked(data, name)
			}
			mem.ChangeFileName(op.fd, name)
			data[name] = op.fd
			m.registerLocked(data, op.fd)

		case opMkdir:
			name := normalizePath(op.path)
			if err := m.lockfreeMkdir(name, op.perm); err != nil {
				return &os.PathError{Op: "mkdir", Path: name, Err: err}
			}

		case opMkdirAll:
			name := normalizePath(op.path)
			// Implicit staging parents use 0o755, matching
			// MemMapFs.registerWithParent.
			for anc := normalizePath(filepath.Dir(name)); anc != name; {
				if anc == FilePathSeparator || anc == "." || anc == "" {
					break
				}
				if err := m.lockfreeMkdir(anc, 0o755); err != nil && data[anc] == nil {
					return &os.PathError{Op: "mkdir", Path: anc, Err: err}
				}
				parent := normalizePath(filepath.Dir(anc))
				if parent == anc {
					break
				}
				anc = parent
			}
			// Target: lockfreeMkdir sets os.ModeDir|perm for new entries;
			// existing entries are left untouched, like MemMapFs.MkdirAll.
			if data[name] == nil {
				if err := m.lockfreeMkdir(name, op.perm); err != nil && data[name] == nil {
					return &os.PathError{Op: "mkdir", Path: name, Err: err}
				}
			}

		case opRemove:
			name := normalizePath(op.path)
			if data[name] == nil {
				return &os.PathError{Op: "remove", Path: name, Err: os.ErrNotExist}
			}
			m.unregisterLocked(data, name)
			delete(data, name)

		case opRemoveAll:
			name := normalizePath(op.path)
			if data[name] != nil {
				m.unregisterLocked(data, name)
			}
			for p := range data {
				if p == name || hasPrefixPath(p, name) {
					delete(data, p)
				}
			}

		case opRename:
			oldName := normalizePath(op.path)
			newName := normalizePath(op.path2)
			src := data[oldName]
			if src == nil {
				return &os.PathError{Op: "rename", Path: oldName, Err: os.ErrNotExist}
			}
			m.unregisterLocked(data, oldName)
			if old := data[newName]; old != nil && old != src {
				m.unregisterLocked(data, newName)
			}
			mem.ChangeFileName(src, newName)
			data[newName] = src
			if err := m.renameDescendants(oldName, newName); err != nil {
				return err
			}
			delete(data, oldName)
			m.registerLocked(data, src)

		case opChmod:
			name := normalizePath(op.path)
			fd := data[name]
			if fd == nil {
				return &os.PathError{Op: "chmod", Path: name, Err: os.ErrNotExist}
			}
			prev := mem.GetFileInfo(fd).Mode() &^ chmodBits
			mem.SetMode(fd, prev|op.perm)

		case opChtimes:
			name := normalizePath(op.path)
			fd := data[name]
			if fd == nil {
				return &os.PathError{Op: "chtimes", Path: name, Err: os.ErrNotExist}
			}
			mem.SetModTime(fd, op.mtime)

		case opChown:
			name := normalizePath(op.path)
			fd := data[name]
			if fd == nil {
				return &os.PathError{Op: "chown", Path: name, Err: os.ErrNotExist}
			}
			mem.SetUID(fd, op.uid)
			mem.SetGID(fd, op.gid)
		}
	}
	return nil
}

// findKeyForLocked returns the key currently holding fd.
func (m *MemMapFs) findKeyForLocked(data map[string]*mem.FileData, fd *mem.FileData) string {
	if fd == nil {
		return ""
	}
	for k, v := range data {
		if v == fd {
			return k
		}
	}
	return ""
}

// unregisterLocked mirrors MemMapFs.unRegisterWithParent but works on the
// supplied map while the caller already holds the write lock.
func (m *MemMapFs) unregisterLocked(data map[string]*mem.FileData, name string) {
	fd := data[name]
	if fd == nil {
		return
	}
	parent := m.findParentLocked(data, fd)
	if parent == nil {
		return
	}
	parent.Lock()
	mem.RemoveFromMemDir(parent, fd)
	parent.Unlock()
}

func (m *MemMapFs) registerLocked(data map[string]*mem.FileData, fd *mem.FileData) {
	parent := m.findParentLocked(data, fd)
	if parent == nil {
		pdir := filepath.Clean(filepath.Dir(fd.Name()))
		if err := m.lockfreeMkdir(pdir, 0o755); err != nil {
			return
		}
		parent = data[pdir]
	}
	if parent == nil {
		return
	}
	parent.Lock()
	mem.InitializeDir(parent)
	mem.AddToMemDir(parent, fd)
	parent.Unlock()
}

func (m *MemMapFs) findParentLocked(data map[string]*mem.FileData, fd *mem.FileData) *mem.FileData {
	pdir := filepath.Clean(filepath.Dir(fd.Name()))
	return data[pdir]
}

func hasPrefixPath(p, prefix string) bool {
	return len(p) > len(prefix) && p[:len(prefix)+1] == prefix+FilePathSeparator
}
