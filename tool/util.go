// Copyright 2019 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package tool

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble"
	"github.com/cockroachdb/pebble/cockroachkvs"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/internal/cache"
	"github.com/cockroachdb/pebble/internal/keyspan"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/vfs"
)

var timeNow = time.Now

type key []byte

func (k *key) String() string {
	return string(*k)
}

func (k *key) Type() string {
	return "key"
}

func (k *key) Set(v string) error {
	switch {
	case strings.HasPrefix(v, "hex:"):
		v = strings.TrimPrefix(v, "hex:")
		b, err := hex.DecodeString(v)
		if err != nil {
			return err
		}
		*k = key(b)

	case strings.HasPrefix(v, "raw:"):
		*k = key(strings.TrimPrefix(v, "raw:"))

	case strings.HasPrefix(v, "crdb:"):
		*k = cockroachkvs.ParseFormattedKey(strings.TrimPrefix(v, "crdb:"))

	default:
		*k = key(v)
	}
	return nil
}

type keyFormatter struct {
	spec      string
	fn        base.FormatKey
	setByUser bool
	comparer  string
}

func (f *keyFormatter) String() string {
	return f.spec
}

func (f *keyFormatter) Type() string {
	return "keyFormatter"
}

func (f *keyFormatter) Set(spec string) error {
	f.spec = spec
	f.setByUser = true
	switch spec {
	case "null":
		f.fn = formatKeyNull
	case "quoted":
		f.fn = formatKeyQuoted
	case "pretty":
		// Using "pretty" defaults to base.FormatBytes (just like formatKeyQuoted),
		// except with the ability of having the comparer-provided formatter
		// overwrite f.fn if there is one specified. We determine whether to do
		// that overwrite through setByUser.
		f.fn = formatKeyQuoted
		f.setByUser = false
	case "size":
		f.fn = formatKeySize
	default:
		if strings.HasPrefix(spec, "pretty:") {
			// Usage: pretty:<comparer-name>
			f.comparer = spec[7:]
			f.fn = formatKeyQuoted
			return nil
		}
		if strings.Count(spec, "%") != 1 {
			return errors.Errorf("unknown formatter: %q", errors.Safe(spec))
		}
		f.fn = func(v []byte) fmt.Formatter {
			return fmtFormatter{f.spec, v}
		}
	}
	return nil
}

func (f *keyFormatter) mustSet(spec string) {
	if err := f.Set(spec); err != nil {
		panic(err)
	}
	f.setByUser = false
}

// Sets the appropriate formatter function for this comparer.
func (f *keyFormatter) setForComparer(comparerName string, comparers sstable.Comparers) {
	if f.setByUser && len(f.comparer) == 0 {
		// User specified a different formatter, no-op.
		return
	}

	if len(f.comparer) > 0 {
		// User specified a comparer to reference for formatting, which takes
		// precedence.
		comparerName = f.comparer
	} else if len(comparerName) == 0 {
		return
	}

	if cmp := comparers[comparerName]; cmp != nil && cmp.FormatKey != nil {
		f.fn = cmp.FormatKey
	}
}

type valueFormatter struct {
	spec      string
	fn        base.FormatValue
	setByUser bool
	comparer  string
}

func (f *valueFormatter) String() string {
	return f.spec
}

func (f *valueFormatter) Type() string {
	return "valueFormatter"
}

func (f *valueFormatter) Set(spec string) error {
	f.spec = spec
	f.setByUser = true
	switch spec {
	case "null":
		f.fn = formatValueNull
	case "quoted":
		f.fn = formatValueQuoted
	case "pretty":
		// Using "pretty" defaults to base.FormatBytes (just like
		// formatValueQuoted), except with the ability of having the
		// comparer-provided formatter overwrite f.fn if there is one specified. We
		// determine whether to do that overwrite through setByUser.
		f.fn = formatValueQuoted
		f.setByUser = false
	case "size":
		f.fn = formatValueSize
	default:
		if strings.HasPrefix(spec, "pretty:") {
			// Usage: pretty:<comparer-name>
			f.comparer = spec[7:]
			f.fn = formatValueQuoted
			return nil
		}
		if strings.Count(spec, "%") != 1 {
			return errors.Errorf("unknown formatter: %q", errors.Safe(spec))
		}
		f.fn = func(k, v []byte) fmt.Formatter {
			return fmtFormatter{f.spec, v}
		}
	}
	fn := f.fn
	// Format blob values differently from inline values.
	f.fn = func(k, v []byte) fmt.Formatter {
		valuePrefix := []byte("blob-value:")
		if bytes.HasPrefix(v, valuePrefix) {
			return fmtFormatter{"[%s]", v[len(valuePrefix):]}
		}
		return fn(k, v)
	}
	return nil
}

func (f *valueFormatter) mustSet(spec string) {
	if err := f.Set(spec); err != nil {
		panic(err)
	}
	f.setByUser = false
}

// Sets the appropriate formatter function for this comparer.
func (f *valueFormatter) setForComparer(comparerName string, comparers sstable.Comparers) {
	if f.setByUser && len(f.comparer) == 0 {
		// User specified a different formatter, no-op.
		return
	}

	if len(f.comparer) > 0 {
		// User specified a comparer to reference for formatting, which takes
		// precedence.
		comparerName = f.comparer
	} else if len(comparerName) == 0 {
		return
	}

	if cmp := comparers[comparerName]; cmp != nil && cmp.FormatValue != nil {
		f.fn = cmp.FormatValue
	}
}

type fmtFormatter struct {
	fmt string
	v   []byte
}

func (f fmtFormatter) Format(s fmt.State, c rune) {
	fmt.Fprintf(s, f.fmt, f.v)
}

type nullFormatter struct{}

func (nullFormatter) Format(s fmt.State, c rune) {
}

func formatKeyNull(v []byte) fmt.Formatter {
	return nullFormatter{}
}

func formatValueNull(k, v []byte) fmt.Formatter {
	return nullFormatter{}
}

func formatKeyQuoted(v []byte) fmt.Formatter {
	return base.FormatBytes(v)
}

func formatValueQuoted(k, v []byte) fmt.Formatter {
	return base.FormatBytes(v)
}

type sizeFormatter []byte

func (v sizeFormatter) Format(s fmt.State, c rune) {
	fmt.Fprintf(s, "<%d>", len(v))
}

func formatKeySize(v []byte) fmt.Formatter {
	return sizeFormatter(v)
}

func formatValueSize(k, v []byte) fmt.Formatter {
	return sizeFormatter(v)
}

func formatKey(w io.Writer, fmtKey keyFormatter, key *base.InternalKey) bool {
	if fmtKey.spec == "null" {
		return false
	}
	fmt.Fprintf(w, "%s", key.Pretty(fmtKey.fn))
	return true
}

func formatSeqNumRange(w io.Writer, start, end base.SeqNum) {
	fmt.Fprintf(w, "<#%d-#%d>", start, end)
}

func formatKeyRange(w io.Writer, fmtKey keyFormatter, start, end *base.InternalKey) {
	if fmtKey.spec == "null" {
		return
	}
	fmt.Fprintf(w, "[%s-%s]", start.Pretty(fmtKey.fn), end.Pretty(fmtKey.fn))
}

func formatKeyValue(
	w io.Writer, fmtKey keyFormatter, fmtValue valueFormatter, key *base.InternalKey, value []byte,
) {
	if key.Kind() == base.InternalKeyKindRangeDelete {
		if fmtKey.spec != "null" {
			fmt.Fprintf(w, "%s-%s#%d,%s",
				fmtKey.fn(key.UserKey), fmtKey.fn(value),
				key.SeqNum(), key.Kind())
		}
	} else {
		needDelimiter := formatKey(w, fmtKey, key)
		if fmtValue.spec != "null" {
			if needDelimiter {
				_, _ = w.Write([]byte{' '})
			}
			fmt.Fprintf(w, "%s", fmtValue.fn(key.UserKey, value))
		}
	}
	_, _ = w.Write([]byte{'\n'})
}

func formatSpan(w io.Writer, fmtKey keyFormatter, fmtValue valueFormatter, s *keyspan.Span) {
	if fmtKey.spec != "null" {
		fmt.Fprintf(w, "[%s-%s):\n", fmtKey.fn(s.Start), fmtKey.fn(s.End))
		for _, k := range s.Keys {
			fmt.Fprintf(w, "  #%d,%s", k.SeqNum(), k.Kind())
			switch k.Kind() {
			case base.InternalKeyKindRangeKeySet:
				fmt.Fprintf(w, ": %s %s", k.Suffix, fmtValue.fn(s.Start, k.Value))
			case base.InternalKeyKindRangeKeyUnset:
				fmt.Fprintf(w, ": %s", k.Suffix)
			}
			_, _ = w.Write([]byte{'\n'})
		}
	}
}

// walk calls fn for each file in dir and its subdirectories (or just on the
// path if it is not a directory),
func walk(stderr io.Writer, fs vfs.FS, path string, fn func(path string)) {
	info, err := fs.Stat(path)
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", path, err)
		return
	}
	if !info.IsDir() {
		fn(path)
		return
	}
	paths, err := fs.List(path)
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", path, err)
		return
	}
	sort.Strings(paths)
	for _, part := range paths {
		walk(stderr, fs, fs.PathJoin(path, part), fn)
	}
}

// processFiles walks through the given arguments and processes files with
// matching extensions using the provided processor.
func processFiles[T any](
	stderr io.Writer,
	fs vfs.FS,
	args []string,
	allowedExts []string,
	newReaderFn func(vfs.File, *cache.Handle, string) (T, error),
	closeReaderFn func(T) error,
	processFileFn func(string, T) error,
) {
	pathFn := func(path string) {
		f, err := fs.Open(path)
		if err != nil {
			fmt.Fprintf(stderr, "%s\n", err)
			return
		}

		// TODO(annie): Use a BufferPool.
		c := pebble.NewCache(128 << 20 /* 128 MB */)
		defer c.Unref()
		ch := c.NewHandle()
		defer ch.Close()

		r, err := newReaderFn(f, ch, path)
		if err != nil {
			fmt.Fprintf(stderr, "%s: %s\n", path, err)
			return
		}
		defer func() {
			_ = closeReaderFn(r)
		}()

		if err := processFileFn(path, r); err != nil {
			fmt.Fprintf(stderr, "%s\n", err)
		}
	}

	// Walk through args and process files with matching extensions.
	for _, arg := range args {
		walk(stderr, fs, arg, func(path string) {
			ext := filepath.Ext(path)
			if slices.Contains(allowedExts, ext) {
				pathFn(path)
			}
		})
	}
}
