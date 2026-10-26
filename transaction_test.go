// Copyright © 2026 Steve Francia <spf@spf13.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

package afero

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

func mustBegin(t *testing.T, fs Fs) (*WriteScope, Fs) {
	t.Helper()
	s, err := BeginWriteScope(fs)
	if err != nil {
		t.Fatalf("BeginWriteScope: %v", err)
	}
	return s, s.View()
}

func scopeWrite(t *testing.T, fs Fs, name, content string) {
	t.Helper()
	f, err := fs.Create(name)
	if err != nil {
		t.Fatalf("create %s: %v", name, err)
	}
	if _, err := f.Write([]byte(content)); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close %s: %v", name, err)
	}
}

func scopeRead(t *testing.T, fs Fs, name string) string {
	t.Helper()
	b, err := ReadFile(fs, name)
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return string(b)
}

func assertNotExist(t *testing.T, fs Fs, name string) {
	t.Helper()
	if _, err := fs.Stat(name); !os.IsNotExist(err) {
		t.Fatalf("expected %s to not exist, got err=%v", name, err)
	}
}

func assertExist(t *testing.T, fs Fs, name string) {
	t.Helper()
	if _, err := fs.Stat(name); err != nil {
		t.Fatalf("expected %s to exist: %v", name, err)
	}
}

// ---------------------------------------------------------------------------
// Basic create / visibility / commit
// ---------------------------------------------------------------------------

func TestWriteScopeCreateVisibleInsideOnly(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/old.txt", "old")

	s, view := mustBegin(t, base)

	// Pre-existing content is visible inside the scope.
	if got := scopeRead(t, view, "/old.txt"); got != "old" {
		t.Fatalf("read old: %q", got)
	}

	scopeWrite(t, view, "/new.txt", "new")

	// Visible inside via Stat/Read/ReadDir/Walk.
	assertExist(t, view, "/new.txt")
	if got := scopeRead(t, view, "/new.txt"); got != "new" {
		t.Fatalf("read new: %q", got)
	}
	names := dirNames(t, view, "/")
	if !containsStr(names, "new.txt") || !containsStr(names, "old.txt") {
		t.Fatalf("scoped readdir: %v", names)
	}
	var walked []string
	_ = Walk(view, "/", func(p string, info os.FileInfo, err error) error {
		walked = append(walked, p)
		return nil
	})
	if !containsStr(walked, filepath.Clean("/new.txt")) && !containsStr(walked, "/new.txt") {
		t.Fatalf("walk did not see new file: %v", walked)
	}

	// Invisible outside while active.
	assertNotExist(t, base, "/new.txt")
	if names := dirNames(t, base, "/"); containsStr(names, "new.txt") {
		t.Fatalf("base readdir leaked: %v", names)
	}

	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertExist(t, base, "/new.txt")
	if got := scopeRead(t, base, "/new.txt"); got != "new" {
		t.Fatalf("post-commit read: %q", got)
	}
}

func TestWriteScopeRollbackLeavesBaseUntouched(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/old.txt", "old")

	s, view := mustBegin(t, base)
	scopeWrite(t, view, "/new.txt", "new")
	if err := view.Rename("/old.txt", "/renamed.txt"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	if err := view.Mkdir("/d", 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	if err := s.Rollback(); err != nil {
		t.Fatalf("rollback: %v", err)
	}

	assertExist(t, base, "/old.txt")
	assertNotExist(t, base, "/new.txt")
	assertNotExist(t, base, "/renamed.txt")
	assertNotExist(t, base, "/d")
	if got := scopeRead(t, base, "/old.txt"); got != "old" {
		t.Fatalf("old content changed: %q", got)
	}

	// Rollback after end is a no-op.
	if err := s.Rollback(); err != nil {
		t.Fatalf("second rollback: %v", err)
	}
}

func TestWriteScopeOverwrite(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "one")

	s, view := mustBegin(t, base)
	f, err := view.OpenFile("/a", os.O_RDWR|os.O_TRUNC, 0o666)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := f.Write([]byte("two")); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if got := scopeRead(t, view, "/a"); got != "two" {
		t.Fatalf("scoped read: %q", got)
	}
	if got := scopeRead(t, base, "/a"); got != "one" {
		t.Fatalf("base leaked: %q", got)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if got := scopeRead(t, base, "/a"); got != "two" {
		t.Fatalf("post-commit: %q", got)
	}
}

func TestWriteScopeDeleteThenCreate(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "old")

	s, view := mustBegin(t, base)
	if err := view.Remove("/a"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	assertNotExist(t, view, "/a")
	assertExist(t, base, "/a")

	scopeWrite(t, view, "/a", "new")
	if got := scopeRead(t, view, "/a"); got != "new" {
		t.Fatalf("scoped read: %q", got)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertExist(t, base, "/a")
	if got := scopeRead(t, base, "/a"); got != "new" {
		t.Fatalf("post-commit: %q", got)
	}
}

func TestWriteScopeDeleteCommitsAbsence(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "x")
	s, view := mustBegin(t, base)
	if err := view.Remove("/a"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertNotExist(t, base, "/a")
}

func TestWriteScopeCreateThenDelete(t *testing.T) {
	base := NewMemMapFs()
	s, view := mustBegin(t, base)
	scopeWrite(t, view, "/a", "x")
	if err := view.Remove("/a"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertNotExist(t, base, "/a")
}

// ---------------------------------------------------------------------------
// Rename / directories / metadata
// ---------------------------------------------------------------------------

func TestWriteScopeRenameFile(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/old", "data")

	s, view := mustBegin(t, base)
	if err := view.Rename("/old", "/new"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	assertNotExist(t, view, "/old")
	if got := scopeRead(t, view, "/new"); got != "data" {
		t.Fatalf("scoped read new: %q", got)
	}
	assertExist(t, base, "/old")

	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertNotExist(t, base, "/old")
	assertExist(t, base, "/new")
	if got := scopeRead(t, base, "/new"); got != "data" {
		t.Fatalf("post-commit: %q", got)
	}
}

func TestWriteScopeRenameDirectoryTree(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/d/f1", "1")
	scopeWrite(t, base, "/d/sub/f2", "2")

	s, view := mustBegin(t, base)
	if err := view.Rename("/d", "/e"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	if got := scopeRead(t, view, "/e/f1"); got != "1" {
		t.Fatalf("e/f1: %q", got)
	}
	if got := scopeRead(t, view, "/e/sub/f2"); got != "2" {
		t.Fatalf("e/sub/f2: %q", got)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertNotExist(t, base, "/d")
	assertExist(t, base, "/e/sub/f2")
	if got := scopeRead(t, base, "/e/sub/f2"); got != "2" {
		t.Fatalf("post-commit: %q", got)
	}
}

func TestWriteScopeMkdirAndReadDirMerge(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/base.txt", "b")

	s, view := mustBegin(t, base)
	if err := view.Mkdir("/newdir", 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	scopeWrite(t, view, "/newdir/inside.txt", "i")

	fi, err := view.Stat("/newdir")
	if err != nil || !fi.IsDir() {
		t.Fatalf("stat newdir: %v %v", fi, err)
	}
	root := dirNames(t, view, "/")
	if !containsStr(root, "newdir") || !containsStr(root, "base.txt") {
		t.Fatalf("merged root: %v", root)
	}
	inside := dirNames(t, view, "/newdir")
	if !containsStr(inside, "inside.txt") {
		t.Fatalf("newdir: %v", inside)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	fi, err = base.Stat("/newdir")
	if err != nil || !fi.IsDir() {
		t.Fatalf("base stat newdir: %v", err)
	}
}

func TestWriteScopeMkdirAllDeep(t *testing.T) {
	base := NewMemMapFs()
	s, view := mustBegin(t, base)
	if err := view.MkdirAll("/x/y/z", 0o750); err != nil {
		t.Fatalf("mkdirall: %v", err)
	}
	assertExist(t, view, "/x/y/z")
	assertNotExist(t, base, "/x")
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertExist(t, base, "/x/y/z")
}

func TestWriteScopeRemoveAll(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/d/a", "1")
	scopeWrite(t, base, "/d/b/c", "2")

	s, view := mustBegin(t, base)
	if err := view.RemoveAll("/d"); err != nil {
		t.Fatalf("removeall: %v", err)
	}
	assertNotExist(t, view, "/d/a")
	assertNotExist(t, view, "/d/b/c")
	assertExist(t, base, "/d/a")
	scopeWrite(t, view, "/d-fresh", "f")
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertNotExist(t, base, "/d")
	assertExist(t, base, "/d-fresh")
}

func TestWriteScopeChmod(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "x")

	s, view := mustBegin(t, base)
	if err := view.Chmod("/a", 0o600); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	fi, _ := view.Stat("/a")
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("scoped mode: %v", fi.Mode())
	}
	fi, _ = base.Stat("/a")
	if fi.Mode().Perm() != 0o666 {
		t.Fatalf("base mode leaked: %v", fi.Mode())
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	fi, _ = base.Stat("/a")
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("post-commit mode: %v", fi.Mode())
	}
}

// ---------------------------------------------------------------------------
// Isolation: parallel scopes and outside readers
// ---------------------------------------------------------------------------

func TestWriteScopeSiblingIsolation(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/base.txt", "b")

	s1, v1 := mustBegin(t, base)
	s2, v2 := mustBegin(t, base)

	scopeWrite(t, v1, "/from1.txt", "1")
	scopeWrite(t, v2, "/from2.txt", "2")

	assertNotExist(t, v1, "/from2.txt")
	assertNotExist(t, v2, "/from1.txt")
	assertExist(t, v1, "/base.txt")
	assertExist(t, v2, "/base.txt")
	assertNotExist(t, base, "/from1.txt")
	assertNotExist(t, base, "/from2.txt")

	if err := s1.Commit(); err != nil {
		t.Fatalf("commit1: %v", err)
	}
	assertExist(t, base, "/from1.txt")
	assertNotExist(t, base, "/from2.txt")
	// v2 never saw s1's UNCOMMITTED content; after the commit it may see
	// the now-committed file through its underlying-Fs fallback.
	assertExist(t, v2, "/base.txt")

	if err := s2.Commit(); err != nil {
		t.Fatalf("commit2: %v", err)
	}
	assertExist(t, base, "/from2.txt")
}

func TestWriteScopeInvisibleToConcurrentOutsideReader(t *testing.T) {
	base := NewMemMapFs()
	s, view := mustBegin(t, base)
	scopeWrite(t, view, "/staged.txt", "s")

	stop := make(chan struct{})
	var committed atomic.Bool
	var leaked atomic.Bool
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				if _, err := base.Stat("/staged.txt"); err == nil && !committed.Load() {
					leaked.Store(true)
					return
				}
			}
		}
	}()

	time.Sleep(50 * time.Millisecond)
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	committed.Store(true)
	close(stop)
	wg.Wait()

	if leaked.Load() {
		t.Fatal("outside reader observed staged content before commit")
	}
	assertExist(t, base, "/staged.txt")
}

// ---------------------------------------------------------------------------
// Nesting
// ---------------------------------------------------------------------------

func TestWriteScopeNestedCommitMerges(t *testing.T) {
	base := NewMemMapFs()

	outer, ov := mustBegin(t, base)
	scopeWrite(t, ov, "/outer.txt", "o")

	inner, iv := mustBegin(t, ov)
	if inner.parent != outer {
		t.Fatal("inner scope did not register outer scope as parent")
	}
	scopeWrite(t, iv, "/inner.txt", "i")
	// The inner scope sees the outer staging; the outer scope must not see
	// uncommitted inner content.
	assertExist(t, iv, "/outer.txt")
	assertNotExist(t, ov, "/inner.txt")
	assertNotExist(t, base, "/inner.txt")
	assertNotExist(t, base, "/outer.txt")

	if err := inner.Commit(); err != nil {
		t.Fatalf("inner commit: %v", err)
	}
	// After inner commit, its staging merges into the outer staging only.
	assertExist(t, ov, "/inner.txt")
	assertNotExist(t, base, "/inner.txt")

	if err := outer.Commit(); err != nil {
		t.Fatalf("outer commit: %v", err)
	}
	assertExist(t, base, "/outer.txt")
	assertExist(t, base, "/inner.txt")
}

func TestWriteScopeNestedInnerRollback(t *testing.T) {
	base := NewMemMapFs()
	outer, ov := mustBegin(t, base)
	scopeWrite(t, ov, "/outer.txt", "o")

	inner, iv := mustBegin(t, ov)
	scopeWrite(t, iv, "/inner.txt", "i")
	if err := inner.Rollback(); err != nil {
		t.Fatalf("inner rollback: %v", err)
	}
	assertNotExist(t, ov, "/inner.txt")
	assertExist(t, ov, "/outer.txt")

	if err := outer.Commit(); err != nil {
		t.Fatalf("outer commit: %v", err)
	}
	assertExist(t, base, "/outer.txt")
	assertNotExist(t, base, "/inner.txt")
}

func TestWriteScopeNestedOuterRollbackDropsInner(t *testing.T) {
	base := NewMemMapFs()
	outer, ov := mustBegin(t, base)

	inner, iv := mustBegin(t, ov)
	scopeWrite(t, iv, "/inner.txt", "i")
	if err := inner.Commit(); err != nil {
		t.Fatalf("inner commit: %v", err)
	}
	assertExist(t, ov, "/inner.txt")

	if err := outer.Rollback(); err != nil {
		t.Fatalf("outer rollback: %v", err)
	}
	assertNotExist(t, base, "/inner.txt")
}

func TestWriteScopeNestedRenameMerge(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/base.txt", "b")

	outer, ov := mustBegin(t, base)
	inner, iv := mustBegin(t, ov)
	if err := iv.Rename("/base.txt", "/renamed.txt"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	if got := scopeRead(t, iv, "/renamed.txt"); got != "b" {
		t.Fatalf("inner read: %q", got)
	}
	if err := inner.Commit(); err != nil {
		t.Fatalf("inner commit: %v", err)
	}
	assertNotExist(t, ov, "/base.txt")
	if got := scopeRead(t, ov, "/renamed.txt"); got != "b" {
		t.Fatalf("outer read: %q", got)
	}
	if err := outer.Commit(); err != nil {
		t.Fatalf("outer commit: %v", err)
	}
	assertNotExist(t, base, "/base.txt")
	if got := scopeRead(t, base, "/renamed.txt"); got != "b" {
		t.Fatalf("base read: %q", got)
	}
}

// ---------------------------------------------------------------------------
// Wrapper semantics
// ---------------------------------------------------------------------------

func TestWriteScopeReadOnlyRejectsWrites(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "x")
	ro := NewReadOnlyFs(base)

	// Sanity: behavior without scope.
	if _, err := ro.Create("/x"); !errors.Is(err, syscall.EPERM) {
		t.Fatalf("plain create err = %v", err)
	}

	s, view := mustBegin(t, ro)
	if _, err := view.Create("/x"); !errors.Is(err, syscall.EPERM) {
		t.Fatalf("scoped create err = %v", err)
	}
	if err := view.Mkdir("/d", 0o755); !errors.Is(err, syscall.EPERM) {
		t.Fatalf("scoped mkdir err = %v", err)
	}
	if err := view.Remove("/a"); !errors.Is(err, syscall.EPERM) {
		t.Fatalf("scoped remove err = %v", err)
	}
	if err := view.Rename("/a", "/b"); !errors.Is(err, syscall.EPERM) {
		t.Fatalf("scoped rename err = %v", err)
	}
	if _, err := view.OpenFile("/a", os.O_WRONLY, 0); !errors.Is(err, syscall.EPERM) {
		t.Fatalf("scoped openfile write err = %v", err)
	}
	// Existing dir MkdirAll is allowed, like ReadOnlyFs.MkdirAll.
	if err := view.MkdirAll("/", 0o755); err != nil {
		t.Fatalf("scoped mkdirall /: %v", err)
	}
	// Reads still work.
	if got := scopeRead(t, view, "/a"); got != "x" {
		t.Fatalf("read through RO scope: %q", got)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertExist(t, base, "/a")
}

func TestWriteScopeBasePathConstraints(t *testing.T) {
	base := NewMemMapFs()
	bp := NewBasePathFs(base, "/base")
	s, view := mustBegin(t, bp)

	scopeWrite(t, view, "rel.txt", "r")
	assertExist(t, view, "rel.txt")
	assertNotExist(t, base, "/base/rel.txt")

	// Paths escaping the base are rejected with the same kind of error as
	// without a scope.
	if _, err := view.Create("../evil.txt"); !os.IsNotExist(err) {
		t.Fatalf("escape create err = %v", err)
	}
	if err := view.Mkdir("../evildir", 0o755); !os.IsNotExist(err) {
		t.Fatalf("escape mkdir err = %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertExist(t, base, "/base/rel.txt")
	assertNotExist(t, base, "/evil.txt")
}

func TestWriteScopeRegexpRules(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/keep.txt", "k")
	re := regexp.MustCompile(`\.txt$`)
	rfs := NewRegexpFs(base, re)

	s, view := mustBegin(t, rfs)
	if _, err := view.Create("/bad.bin"); !errors.Is(err, syscall.ENOENT) {
		t.Fatalf("create bad.bin: %v", err)
	}
	scopeWrite(t, view, "/good.txt", "g")
	// The filter also applies to renames of files.
	if err := view.Rename("/keep.txt", "/renamed.bin"); !errors.Is(err, syscall.ENOENT) {
		t.Fatalf("rename to non-match: %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertExist(t, base, "/good.txt")
	assertExist(t, base, "/keep.txt")
}

func TestWriteScopeCacheOnRead(t *testing.T) {
	base := NewMemMapFs()
	layer := NewMemMapFs()
	union := NewCacheOnReadFs(base, layer, 0)

	s, view := mustBegin(t, union)
	scopeWrite(t, view, "/cached-write.txt", "c")
	assertNotExist(t, base, "/cached-write.txt")
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertExist(t, base, "/cached-write.txt")
	assertExist(t, layer, "/cached-write.txt")
}

// ---------------------------------------------------------------------------
// Open handle safety
// ---------------------------------------------------------------------------

func TestWriteScopeUnclosedWriteHandleCommit(t *testing.T) {
	base := NewMemMapFs()
	s, view := mustBegin(t, base)

	f, err := view.Create("/a")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if _, err := f.Write([]byte("partial")); err != nil {
		t.Fatalf("write: %v", err)
	}
	// Deliberately do not Close before commit.
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if got := scopeRead(t, base, "/a"); got != "partial" {
		t.Fatalf("base content: %q", got)
	}
	// The stale handle can no longer mutate the base.
	if _, err := f.Write([]byte("LATE")); !errors.Is(err, ErrFileClosed) {
		t.Fatalf("post-commit write err = %v", err)
	}
	if err := f.Truncate(0); !errors.Is(err, ErrFileClosed) {
		t.Fatalf("post-commit truncate err = %v", err)
	}
	if got := scopeRead(t, base, "/a"); got != "partial" {
		t.Fatalf("base changed by stale handle: %q", got)
	}
}

func TestWriteScopeUnclosedWriteHandleRollback(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "base")
	s, view := mustBegin(t, base)

	f, err := view.OpenFile("/a", os.O_RDWR, 0o666)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := f.Write([]byte("xx")); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := s.Rollback(); err != nil {
		t.Fatalf("rollback: %v", err)
	}
	if got := scopeRead(t, base, "/a"); got != "base" {
		t.Fatalf("base changed: %q", got)
	}
	if _, err := f.Write([]byte("yy")); !errors.Is(err, ErrFileClosed) {
		t.Fatalf("post-rollback write err = %v", err)
	}
}

func TestWriteScopeDeleteThenExclusiveCreate(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "old")

	s, view := mustBegin(t, base)
	if err := view.Remove("/a"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	f, err := view.OpenFile("/a", os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatalf("excl create after delete: %v", err)
	}
	if _, err := f.Write([]byte("fresh")); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if got := scopeRead(t, base, "/a"); got != "fresh" {
		t.Fatalf("post-commit: %q", got)
	}
	fi, _ := base.Stat("/a")
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("perm: %v", fi.Mode())
	}
}

func TestWriteScopeRenameStagedOnlyFile(t *testing.T) {
	base := NewMemMapFs()
	s, view := mustBegin(t, base)
	scopeWrite(t, view, "/a", "data")
	if err := view.Rename("/a", "/b"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	assertNotExist(t, view, "/a")
	if got := scopeRead(t, view, "/b"); got != "data" {
		t.Fatalf("scoped read: %q", got)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertNotExist(t, base, "/a")
	if got := scopeRead(t, base, "/b"); got != "data" {
		t.Fatalf("post-commit: %q", got)
	}
}

func TestWriteScopeWriteThroughHandleAfterRename(t *testing.T) {
	base := NewMemMapFs()
	s, view := mustBegin(t, base)
	f, err := view.Create("/a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte("first")); err != nil {
		t.Fatal(err)
	}
	if err := view.Rename("/a", "/b"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	// Keep writing through the original handle; bytes belong to /b now.
	if _, err := f.Write([]byte("-second")); err != nil {
		t.Fatalf("write after rename: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertNotExist(t, base, "/a")
	if got := scopeRead(t, base, "/b"); got != "first-second" {
		t.Fatalf("post-commit: %q", got)
	}
}

func TestWriteScopeAppendSeesBaseContent(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "base")
	s, view := mustBegin(t, base)
	f, err := view.OpenFile("/a", os.O_RDWR|os.O_APPEND, 0o666)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte("+appended")); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if got := scopeRead(t, view, "/a"); got != "base+appended" {
		t.Fatalf("scoped: %q", got)
	}
	if got := scopeRead(t, base, "/a"); got != "base" {
		t.Fatalf("base leaked: %q", got)
	}
	if err := s.Commit(); err != nil {
		t.Fatal(err)
	}
	if got := scopeRead(t, base, "/a"); got != "base+appended" {
		t.Fatalf("post-commit: %q", got)
	}
}

func TestWriteScopeRemoveNonEmptyDirFails(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/d/f", "x")
	s, view := mustBegin(t, base)
	if err := view.Remove("/d"); !errors.Is(err, syscall.ENOTEMPTY) {
		t.Fatalf("remove non-empty dir: %v", err)
	}
	if err := view.RemoveAll("/d"); err != nil {
		t.Fatalf("removeall: %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertNotExist(t, base, "/d")
}

func TestWriteScopeNestedModifiesOuterStagedFile(t *testing.T) {
	base := NewMemMapFs()
	outer, ov := mustBegin(t, base)
	scopeWrite(t, ov, "/a", "outer")

	inner, iv := mustBegin(t, ov)
	f, err := iv.OpenFile("/a", os.O_RDWR|os.O_TRUNC, 0o666)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte("inner")); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if got := scopeRead(t, iv, "/a"); got != "inner" {
		t.Fatalf("inner read: %q", got)
	}
	if got := scopeRead(t, ov, "/a"); got != "outer" {
		t.Fatalf("outer must not see uncommitted inner overwrite: %q", got)
	}
	if err := inner.Commit(); err != nil {
		t.Fatal(err)
	}
	if got := scopeRead(t, ov, "/a"); got != "inner" {
		t.Fatalf("outer read after inner commit: %q", got)
	}
	if err := outer.Commit(); err != nil {
		t.Fatal(err)
	}
	if got := scopeRead(t, base, "/a"); got != "inner" {
		t.Fatalf("base read: %q", got)
	}
}

func TestWriteScopeOuterEndsWhileInnerActive(t *testing.T) {
	base := NewMemMapFs()
	outer, ov := mustBegin(t, base)
	inner, _ := mustBegin(t, ov)

	if err := outer.Commit(); !errors.Is(err, errInnerScopeActive) {
		t.Fatalf("outer commit with active inner: %v", err)
	}
	if err := outer.Rollback(); !errors.Is(err, errInnerScopeActive) {
		t.Fatalf("outer rollback with active inner: %v", err)
	}

	if err := inner.Rollback(); err != nil {
		t.Fatalf("inner rollback: %v", err)
	}
	if err := outer.Rollback(); err != nil {
		t.Fatalf("outer rollback after inner ends: %v", err)
	}
}

func TestWriteScopeNestedInnerDeletesBaseFile(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "x")
	outer, ov := mustBegin(t, base)
	inner, iv := mustBegin(t, ov)
	if err := iv.Remove("/a"); err != nil {
		t.Fatal(err)
	}
	assertNotExist(t, iv, "/a")
	assertExist(t, ov, "/a") // uncommitted inner delete hidden from outer
	if err := inner.Commit(); err != nil {
		t.Fatal(err)
	}
	assertNotExist(t, ov, "/a")
	assertExist(t, base, "/a")
	if err := outer.Commit(); err != nil {
		t.Fatal(err)
	}
	assertNotExist(t, base, "/a")
}

// ---------------------------------------------------------------------------
// Scope object ownership
// ---------------------------------------------------------------------------

func TestWriteScopeReentryAndReuse(t *testing.T) {
	base := NewMemMapFs()
	s := NewWriteScope(base)
	view, err := s.Enter()
	if err != nil {
		t.Fatalf("enter: %v", err)
	}
	if _, err := s.Enter(); !errors.Is(err, ErrScopeReentry) {
		t.Fatalf("re-entry err = %v", err)
	}
	scopeWrite(t, view, "/a", "a")
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if err := s.Commit(); !errors.Is(err, ErrScopeInactive) {
		t.Fatalf("double commit err = %v", err)
	}
	// Binding is gone: the same object can be entered again.
	view2, err := s.Enter()
	if err != nil {
		t.Fatalf("re-enter: %v", err)
	}
	assertExist(t, view2, "/a")
	if err := s.Rollback(); err != nil {
		t.Fatalf("rollback: %v", err)
	}
}

func TestWriteScopeBusyFromAnotherTask(t *testing.T) {
	base := NewMemMapFs()
	s := NewWriteScope(base)
	if _, err := s.Enter(); err != nil {
		t.Fatalf("enter: %v", err)
	}

	errCh := make(chan error, 1)
	go func() {
		_, err := s.Enter()
		errCh <- err
	}()
	select {
	case err := <-errCh:
		if !errors.Is(err, ErrScopeBusy) {
			t.Fatalf("busy err = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("other task did not receive busy error")
	}

	commitErr := make(chan error, 1)
	go func() { commitErr <- s.Commit() }()
	select {
	case err := <-commitErr:
		if !errors.Is(err, ErrScopeOwner) {
			t.Fatalf("foreign commit err = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("foreign commit did not return")
	}

	if err := s.Rollback(); err != nil {
		t.Fatalf("rollback: %v", err)
	}
}

// ---------------------------------------------------------------------------
// OsFs-backed scope (generic replay)
// ---------------------------------------------------------------------------

func TestWriteScopeOnOsFs(t *testing.T) {
	dir := t.TempDir()
	fs := NewOsFs()

	p := func(name string) string { return filepath.Join(dir, name) }
	if err := WriteFile(fs, p("old.txt"), []byte("old"), 0o666); err != nil {
		t.Fatal(err)
	}
	if err := fs.Mkdir(p("d"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(fs, p("d/keep.txt"), []byte("k"), 0o666); err != nil {
		t.Fatal(err)
	}

	s, view := mustBegin(t, fs)
	scopeWrite(t, view, p("new.txt"), "new")
	// Rename into a directory that does not exist yet; it is created at
	// commit time.
	if err := view.Rename(p("old.txt"), p("nested/renamed.txt")); err != nil {
		t.Fatalf("rename: %v", err)
	}
	if err := view.Remove(p("d/keep.txt")); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if _, err := view.Stat(p("new.txt")); err != nil {
		t.Fatalf("stat staged: %v", err)
	}
	if _, err := fs.Stat(p("new.txt")); !os.IsNotExist(err) {
		t.Fatalf("base leaked new.txt: %v", err)
	}

	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if b, err := ReadFile(fs, p("new.txt")); err != nil || string(b) != "new" {
		t.Fatalf("new.txt: %q %v", b, err)
	}
	if _, err := fs.Stat(p("old.txt")); !os.IsNotExist(err) {
		t.Fatalf("old.txt still there: %v", err)
	}
	if b, err := ReadFile(fs, p("nested/renamed.txt")); err != nil || string(b) != "old" {
		t.Fatalf("renamed.txt: %q %v", b, err)
	}
	if _, err := fs.Stat(p("d/keep.txt")); !os.IsNotExist(err) {
		t.Fatalf("keep.txt not removed: %v", err)
	}
}

func TestWriteScopeOnBasePathMemMap(t *testing.T) {
	base := NewMemMapFs()
	if err := base.Mkdir("/root", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(base, "/root/existing.txt", []byte("e"), 0o666); err != nil {
		t.Fatal(err)
	}
	bp := NewBasePathFs(base, "/root")

	s, view := mustBegin(t, bp)
	scopeWrite(t, view, "new.txt", "n")
	if err := view.Rename("existing.txt", "moved.txt"); err != nil {
		t.Fatal(err)
	}
	if err := view.Mkdir("d", 0o750); err != nil {
		t.Fatal(err)
	}
	// Not visible outside the scope before commit.
	assertNotExist(t, base, "/root/new.txt")
	assertExist(t, base, "/root/existing.txt")

	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	assertExist(t, base, "/root/new.txt")
	assertExist(t, base, "/root/moved.txt")
	assertNotExist(t, base, "/root/existing.txt")
	fi, err := base.Stat("/root/d")
	if err != nil || !fi.IsDir() {
		t.Fatalf("dir missing: %v %v", fi, err)
	}
	if fi.Mode().Perm() != 0o750 {
		t.Fatalf("dir perm: %v", fi.Mode())
	}
	// Nothing escaped the base path.
	names := dirNames(t, base, "/")
	if containsStr(names, "new.txt") || containsStr(names, "moved.txt") {
		t.Fatalf("files escaped base path: %v", names)
	}
}

func TestWriteScopeOnBasePathOsFs(t *testing.T) {
	dir := t.TempDir()
	bp := NewBasePathFs(NewOsFs(), dir)

	s, view := mustBegin(t, bp)
	scopeWrite(t, view, "new.txt", "n")
	if err := view.MkdirAll("sub/deep", 0o755); err != nil {
		t.Fatal(err)
	}
	scopeWrite(t, view, "sub/deep/f.txt", "f")
	if _, err := bp.Stat("new.txt"); !os.IsNotExist(err) {
		t.Fatalf("leaked before commit: %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if b, err := ReadFile(bp, "new.txt"); err != nil || string(b) != "n" {
		t.Fatalf("new.txt: %q %v", b, err)
	}
	if _, err := bp.Stat("sub/deep/f.txt"); err != nil {
		t.Fatalf("deep file missing: %v", err)
	}
}

func TestWriteScopeChtimes(t *testing.T) {
	base := NewMemMapFs()
	scopeWrite(t, base, "/a", "x")
	mt := time.Date(2020, 1, 2, 3, 4, 5, 0, time.UTC)

	s, view := mustBegin(t, base)
	if err := view.Chtimes("/a", mt, mt); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
	fi, _ := view.Stat("/a")
	if !fi.ModTime().Equal(mt) {
		t.Fatalf("scoped mtime: %v", fi.ModTime())
	}
	if err := s.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
	fi, _ = base.Stat("/a")
	if !fi.ModTime().Equal(mt) {
		t.Fatalf("base mtime: %v", fi.ModTime())
	}
}

func TestWriteScopeOnOsFsRollback(t *testing.T) {
	dir := t.TempDir()
	fs := NewOsFs()
	p := func(name string) string { return filepath.Join(dir, name) }
	if err := WriteFile(fs, p("old.txt"), []byte("old"), 0o666); err != nil {
		t.Fatal(err)
	}

	s, view := mustBegin(t, fs)
	scopeWrite(t, view, p("new.txt"), "new")
	if err := view.Remove(p("old.txt")); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if err := s.Rollback(); err != nil {
		t.Fatalf("rollback: %v", err)
	}
	if _, err := fs.Stat(p("new.txt")); !os.IsNotExist(err) {
		t.Fatalf("new.txt leaked after rollback")
	}
	if b, err := ReadFile(fs, p("old.txt")); err != nil || string(b) != "old" {
		t.Fatalf("old.txt changed: %q %v", b, err)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func dirNames(t *testing.T, fs Fs, dir string) []string {
	t.Helper()
	f, err := fs.Open(dir)
	if err != nil {
		t.Fatalf("open dir %s: %v", dir, err)
	}
	list, err := f.Readdir(-1)
	_ = f.Close()
	if err != nil && !errors.Is(err, io.EOF) {
		t.Fatalf("readdir %s: %v", dir, err)
	}
	names := make([]string, 0, len(list))
	for _, fi := range list {
		names = append(names, fi.Name())
	}
	sort.Strings(names)
	return names
}

func containsStr(list []string, v string) bool {
	for _, x := range list {
		if x == v {
			return true
		}
	}
	return false
}
