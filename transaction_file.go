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
	"io"
	iofs "io/fs"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/spf13/afero/internal/common"
	"github.com/spf13/afero/mem"
)

// txnRWFile is a write-capable scope handle. It always operates on private
// staging bytes; it never wraps a writable handle to the underlying Fs.
// After the scope ends, mutating calls fail so rolled-back bytes can never
// write through and a committed file can never be half-rewritten by a
// leftover handle.
type txnRWFile struct {
	f  *mem.File
	fd *mem.FileData
	t  *txnFs

	mu     sync.Mutex
	closed bool
	lazy   bool // journal the first mutation instead of creation
	dirty  bool
}

func (t *txnFs) newRWFile(f *mem.File, fd *mem.FileData, lazy bool) *txnRWFile {
	return &txnRWFile{f: f, fd: fd, t: t, lazy: lazy}
}

func (h *txnRWFile) ended() bool {
	return atomic.LoadInt32(h.t.state) != scopeStateActive
}

// noteWrite appends the write journal entry on the first actual mutation.
// The entry records the FileData identity; renames move that identity, so a
// later replay lands the bytes on the right path in the right order.
func (h *txnRWFile) noteWrite() error {
	if h.ended() {
		return ErrFileClosed
	}
	if h.lazy && !h.dirty {
		h.dirty = true
		h.t.journal = append(h.t.journal, txOp{
			kind: opWrite,
			path: h.fd.Name(),
			fd:   h.fd,
		})
	}
	return nil
}

func (h *txnRWFile) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return ErrFileClosed
	}
	h.closed = true
	if h.ended() {
		// Do not touch FileData that now belongs to (or never reached) the
		// underlying Fs.
		return nil
	}
	return h.f.Close()
}

func (h *txnRWFile) Read(b []byte) (int, error) { return h.f.Read(b) }
func (h *txnRWFile) ReadAt(b []byte, off int64) (int, error) {
	return h.f.ReadAt(b, off)
}
func (h *txnRWFile) Seek(offset int64, whence int) (int64, error) {
	return h.f.Seek(offset, whence)
}
func (h *txnRWFile) Sync() error  { return h.f.Sync() }
func (h *txnRWFile) Name() string { return h.f.Name() }
func (h *txnRWFile) Stat() (os.FileInfo, error) {
	return h.f.Stat()
}
func (h *txnRWFile) Readdir(count int) ([]os.FileInfo, error) {
	return h.f.Readdir(count)
}
func (h *txnRWFile) Readdirnames(n int) ([]string, error) {
	return h.f.Readdirnames(n)
}
func (h *txnRWFile) ReadDir(n int) ([]iofs.DirEntry, error) {
	return h.f.ReadDir(n)
}

func (h *txnRWFile) Write(b []byte) (int, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := h.noteWrite(); err != nil {
		return 0, err
	}
	return h.f.Write(b)
}

func (h *txnRWFile) WriteAt(b []byte, off int64) (int, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := h.noteWrite(); err != nil {
		return 0, err
	}
	return h.f.WriteAt(b, off)
}

func (h *txnRWFile) WriteString(s string) (int, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := h.noteWrite(); err != nil {
		return 0, err
	}
	return h.f.Write([]byte(s))
}

func (h *txnRWFile) Truncate(size int64) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := h.noteWrite(); err != nil {
		return err
	}
	return h.f.Truncate(size)
}

// txnDirFile is a read-only directory handle listing the merged
// underlying + staging contents.
type txnDirFile struct {
	t    *txnFs
	name string
	key  string

	mu      sync.Mutex
	once    sync.Once
	entries []os.FileInfo
	off     int
	closed  bool
	fi      os.FileInfo
}

func (t *txnFs) newDirHandle(name, key string, overlayFd *mem.FileData) File {
	return &txnDirFile{t: t, name: name, key: key}
}

func (d *txnDirFile) build() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.buildLocked()
}

func (d *txnDirFile) buildLocked() {
	entriesByName := map[string]os.FileInfo{}

	// Underlying entries.
	baseFi, baseErr := d.t.top.Stat(d.name)
	baseDir := baseErr == nil && baseFi.IsDir()
	if baseDir {
		if f, err := d.t.top.Open(d.name); err == nil {
			if list, err := f.Readdir(-1); err == nil {
				for _, fi := range list {
					entriesByName[fi.Name()] = fi
				}
			}
			_ = f.Close()
		}
	}

	// Staging entries shadow underlying entries with the same child name;
	// implicit staging dirs defer to the underlying entry when present.
	for _, fd := range d.t.overlayChildren(d.key) {
		base := filepath.Base(fd.Name())
		sfi := mem.GetFileInfo(fd)
		if _, implicit := d.t.implicit[fd]; implicit {
			if _, exists := entriesByName[base]; exists {
				continue
			}
		}
		entriesByName[base] = sfi
	}

	// Apply tombstones to underlying entries.
	for name := range entriesByName {
		childKey := normalizePath(filepath.Join(d.key, name))
		if _, del := d.t.deleted[childKey]; del {
			delete(entriesByName, name)
		}
	}

	out := make([]os.FileInfo, 0, len(entriesByName))
	for _, fi := range entriesByName {
		out = append(out, fi)
	}
	sort.Sort(byName(out))
	d.entries = out

	if fd, ok := d.t.overlayGet(d.key); ok {
		if _, implicit := d.t.implicit[fd]; !implicit {
			d.fi = mem.GetFileInfo(fd)
		}
	}
	if d.fi == nil {
		if baseDir {
			d.fi = baseFi
		} else if fd, ok := d.t.overlayGet(d.key); ok {
			d.fi = mem.GetFileInfo(fd)
		} else {
			d.fi = nil
		}
	}
}

func (d *txnDirFile) ensure() error {
	if atomic.LoadInt32(d.t.state) != scopeStateActive {
		return ErrFileClosed
	}
	d.mu.Lock()
	closed := d.closed
	d.mu.Unlock()
	if closed {
		return ErrFileClosed
	}
	d.once.Do(func() { d.build() })
	return nil
}

func (d *txnDirFile) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.closed = true
	return nil
}

func (d *txnDirFile) Name() string { return d.name }

func (d *txnDirFile) Stat() (os.FileInfo, error) {
	if err := d.ensure(); err != nil {
		return nil, err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.fi == nil {
		return nil, notExist("stat", d.name)
	}
	return d.fi, nil
}

func (d *txnDirFile) Readdir(count int) ([]os.FileInfo, error) {
	if err := d.ensure(); err != nil {
		return nil, err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.off >= len(d.entries) {
		if count > 0 {
			return nil, io.EOF
		}
		return nil, nil
	}
	var err error
	end := len(d.entries)
	if count > 0 {
		end = d.off + count
		if end > len(d.entries) {
			end = len(d.entries)
		}
		if end >= len(d.entries) {
			err = io.EOF
		}
	}
	out := make([]os.FileInfo, end-d.off)
	copy(out, d.entries[d.off:end])
	d.off = end
	return out, err
}

func (d *txnDirFile) Readdirnames(n int) ([]string, error) {
	fi, err := d.Readdir(n)
	names := make([]string, len(fi))
	for i, f := range fi {
		names[i] = f.Name()
	}
	return names, err
}

func (d *txnDirFile) ReadDir(n int) ([]iofs.DirEntry, error) {
	items, err := d.Readdir(n)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	out := make([]iofs.DirEntry, len(items))
	for i := range items {
		out[i] = common.FileInfoDirEntry{FileInfo: items[i]}
	}
	if err != nil {
		return out, err
	}
	return out, nil
}

func (d *txnDirFile) Read([]byte) (int, error) {
	return 0, io.EOF
}

func (d *txnDirFile) ReadAt([]byte, int64) (int, error) {
	return 0, io.EOF
}

func (d *txnDirFile) Seek(int64, int) (int64, error) { return 0, nil }

func (d *txnDirFile) Write([]byte) (int, error) {
	return 0, &os.PathError{Op: "write", Path: d.name, Err: errors.New("file handle is read only")}
}

func (d *txnDirFile) WriteAt([]byte, int64) (int, error) {
	return 0, &os.PathError{Op: "write", Path: d.name, Err: errors.New("file handle is read only")}
}

func (d *txnDirFile) WriteString(string) (int, error) {
	return 0, &os.PathError{Op: "write", Path: d.name, Err: errors.New("file handle is read only")}
}

func (d *txnDirFile) Truncate(int64) error {
	return &os.PathError{Op: "truncate", Path: d.name, Err: errors.New("file handle is read only")}
}

func (d *txnDirFile) Sync() error { return nil }
