// Copyright 2013 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package pebble

import (
	"bytes"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cockroachdb/datadriven"
	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/internal/bytealloc"
	"github.com/cockroachdb/pebble/internal/cache"
	"github.com/cockroachdb/pebble/internal/invalidating"
	"github.com/cockroachdb/pebble/internal/manifest"
	"github.com/cockroachdb/pebble/internal/testkeys"
	"github.com/cockroachdb/pebble/objstorage/objstorageprovider"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/sstable/block/blockkind"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
	"golang.org/x/sync/errgroup"
)

var testKeyValuePairs = []string{
	"10:10",
	"11:11",
	"12:12",
	"13:13",
	"14:14",
	"15:15",
	"16:16",
	"17:17",
	"18:18",
	"19:19",
}

// testIterator tests creating a combined iterator from a number of sub-
// iterators. newFunc is a constructor function. splitFunc returns a random
// split of the testKeyValuePairs slice such that walking a combined iterator
// over those splits should recover the original key/value pairs in order.
func testIterator(
	t *testing.T,
	newFunc func(...internalIterator) internalIterator,
	splitFunc func(r *rand.Rand) [][]string,
) {
	fakeIterWithCloseErr := func(kvs []base.InternalKV, errorMsg string) *base.FakeIter {
		f := base.NewFakeIter(kvs)
		f.SetCloseErr(errors.New(errorMsg))
		return f
	}
	// Test pre-determined sub-iterators. The sub-iterators are designed
	// so that the combined key/value pair order is the same whether the
	// combined iterator is concatenating or merging.
	testCases := []struct {
		desc  string
		iters []internalIterator
		want  string
	}{
		{
			"one sub-iterator",
			[]internalIterator{
				base.NewFakeIter(base.FakeKVs("e:1", "w:2")),
			},
			"<e:1><w:2>.",
		},
		{
			"two sub-iterators",
			[]internalIterator{
				base.NewFakeIter(base.FakeKVs("a0:0")),
				base.NewFakeIter(base.FakeKVs("b1:1", "b2:2")),
			},
			"<a0:0><b1:1><b2:2>.",
		},
		{
			"empty sub-iterators",
			[]internalIterator{
				base.NewFakeIter(nil),
				base.NewFakeIter(nil),
				base.NewFakeIter(nil),
			},
			".",
		},
		{
			"sub-iterator errors",
			[]internalIterator{
				base.NewFakeIter(base.FakeKVs("a0:0", "a1:1")),
				fakeIterWithCloseErr(base.FakeKVs("b2:2", "b3:3", "b4:4"), "the sky is falling"),
				fakeIterWithCloseErr(base.FakeKVs("c5:5", "c6:6"), "run for your lives"),
			},
			"<a0:0><a1:1><b2:2><b3:3><b4:4>err=the sky is falling",
		},
	}
	for _, tc := range testCases {
		var b bytes.Buffer
		iter := invalidating.NewIter(newFunc(tc.iters...))
		for kv := iter.First(); kv != nil; kv = iter.Next() {
			fmt.Fprintf(&b, "<%s:%d>", kv.K.UserKey, kv.SeqNum())
		}
		if err := iter.Close(); err != nil {
			fmt.Fprintf(&b, "err=%v", err)
		} else {
			b.WriteByte('.')
		}
		if got := b.String(); got != tc.want {
			t.Errorf("%s:\ngot  %q\nwant %q", tc.desc, got, tc.want)
		}
	}

	// Test randomly generated sub-iterators.
	r := rand.New(rand.NewPCG(0, 0))
	for i, nBad := 0, 0; i < 1000; i++ {
		bad := false

		splits := splitFunc(r)
		iters := make([]internalIterator, len(splits))
		for i, split := range splits {
			iters[i] = base.NewFakeIter(base.FakeKVs(split...))
		}
		iter := invalidating.NewIter(newFunc(iters...))
		kv := iter.First()
		j := 0
		for ; kv != nil && j < len(testKeyValuePairs); j++ {
			got := fmt.Sprintf("%s:%d", kv.K.UserKey, kv.SeqNum())
			want := testKeyValuePairs[j]
			if got != want {
				bad = true
				t.Errorf("random splits: i=%d, j=%d: got %q, want %q", i, j, got, want)
			}
			kv = iter.Next()
		}
		if kv != nil {
			bad = true
			t.Errorf("random splits: i=%d, j=%d: iter was not exhausted", i, j)
		}
		if j != len(testKeyValuePairs) {
			bad = true
			t.Errorf("random splits: i=%d, j=%d: want j=%d", i, j, len(testKeyValuePairs))
			return
		}
		if err := iter.Close(); err != nil {
			bad = true
			t.Errorf("random splits: i=%d, j=%d: %v", i, j, err)
		}

		if bad {
			nBad++
			if nBad == 10 {
				t.Fatal("random splits: too many errors; stopping")
			}
		}
	}
}

func TestIterator(t *testing.T) {
	var merge Merge
	var kvs []base.InternalKV

	newIter := func(seqNum base.SeqNum, opts IterOptions) *Iterator {
		if merge == nil {
			merge = DefaultMerger.Merge
		}
		wrappedMerge := func(key, value []byte) (ValueMerger, error) {
			if len(key) == 0 {
				t.Fatalf("an empty key is passed into Merge")
			}
			return merge(key, value)
		}
		it := &Iterator{
			opts:     opts,
			comparer: *testkeys.Comparer,
			merge:    wrappedMerge,
		}
		// NB: Use a mergingIter to filter entries newer than seqNum.
		fakeIter := base.NewFakeIter(kvs)
		fakeIter.SetBounds(opts.GetLowerBound(), opts.GetUpperBound())
		iter := newMergingIter(nil /* logger */, &it.stats.InternalStats, it.cmp, it.comparer.Split, fakeIter)
		iter.snapshot = seqNum
		// NB: This Iterator cannot be cloned since it is not constructed
		// with a readState. It suffices for this test.
		it.iter = invalidating.NewIter(iter)
		return it
	}

	datadriven.RunTest(t, "testdata/iterator", func(t *testing.T, d *datadriven.TestData) string {
		switch d.Cmd {
		case "define":
			merge = nil
			if arg, ok := d.Arg("merger"); ok && len(arg.Vals[0]) > 0 && arg.Vals[0] == "deletable" {
				merge = base.NewDeletableSumValueMerger
			}
			kvs = kvs[:0]
			for _, key := range strings.Split(d.Input, "\n") {
				j := strings.Index(key, ":")
				kvs = append(kvs, base.MakeInternalKV(base.ParseInternalKey(key[:j]), []byte(key[j+1:])))
			}
			return ""

		case "iter":
			var seqNum uint64
			var opts IterOptions
			d.MaybeScanArgs(t, "seq", &seqNum)
			var lower, upper string
			if d.MaybeScanArgs(t, "lower", &lower) {
				opts.LowerBound = []byte(lower)
			}
			if d.MaybeScanArgs(t, "upper", &upper) {
				opts.UpperBound = []byte(upper)
			}

			iter := newIter(base.SeqNum(seqNum), opts)
			iterOutput := runIterCmd(d, iter, true)
			stats := iter.Stats()
			return fmt.Sprintf("%sstats: %s\n", iterOutput, stats.String())

		default:
			return fmt.Sprintf("unknown command: %s", d.Cmd)
		}
	})
}

type minSeqNumPropertyCollector struct {
	minSeqNum base.SeqNum
}

var _ BlockPropertyCollector = (*minSeqNumPropertyCollector)(nil)

func (c *minSeqNumPropertyCollector) Name() string {
	return "minSeqNumPropertyCollector"
}

func (c *minSeqNumPropertyCollector) AddPointKey(key InternalKey, value []byte) error {
	if c.minSeqNum == 0 || c.minSeqNum > key.SeqNum() {
		c.minSeqNum = key.SeqNum()
	}
	return nil
}

func (c *minSeqNumPropertyCollector) AddRangeKeys(span sstable.Span) error {
	for _, k := range span.Keys {
		if c.minSeqNum == 0 || c.minSeqNum > k.SeqNum() {
			c.minSeqNum = k.SeqNum()
		}
	}
	return nil
}

func (c *minSeqNumPropertyCollector) FinishDataBlock(buf []byte) ([]byte, error) {
	return nil, nil
}

func (c *minSeqNumPropertyCollector) AddPrevDataBlockToIndexBlock() {}

func (c *minSeqNumPropertyCollector) FinishIndexBlock(buf []byte) ([]byte, error) {
	return nil, nil
}

func (c *minSeqNumPropertyCollector) FinishTable(buf []byte) ([]byte, error) {
	return binary.AppendUvarint(buf, uint64(c.minSeqNum)), nil
}

func (c *minSeqNumPropertyCollector) AddCollectedWithSuffixReplacement(
	oldProp []byte, oldSuffix, newSuffix []byte,
) error {
	return errors.Errorf("not implemented")
}

func (c *minSeqNumPropertyCollector) SupportsSuffixReplacement() bool {
	return false
}

// minSeqNumFilter is a BlockPropertyFilter that uses the
// minSeqNumPropertyCollector data to filter out entire tables.
type minSeqNumFilter struct {
	seqNumUpperBound uint64
}

var _ BlockPropertyFilter = (*minSeqNumFilter)(nil)

func (*minSeqNumFilter) Name() string {
	return (&minSeqNumPropertyCollector{}).Name()
}

func (f *minSeqNumFilter) Intersects(prop []byte) (bool, error) {
	// Blocks will have no data.
	if len(prop) == 0 {
		return true, nil
	}
	minSeqNum, n := binary.Uvarint(prop)
	if n <= 0 {
		return false, errors.Errorf("invalid block property data %v", prop)
	}
	return minSeqNum < f.seqNumUpperBound, nil
}

func (f *minSeqNumFilter) SyntheticSuffixIntersects(prop []byte, suffix []byte) (bool, error) {
	panic("unimplemented")
}

func TestReadSampling(t *testing.T) {
	var d *DB
	defer func() {
		if d != nil {
			require.NoError(t, d.Close())
		}
	}()

	var iter *Iterator
	defer func() {
		if iter != nil {
			require.NoError(t, iter.Close())
		}
	}()

	datadriven.RunTest(t, "testdata/iterator_read_sampling", func(t *testing.T, td *datadriven.TestData) string {
		switch td.Cmd {
		case "define":
			if iter != nil {
				if err := iter.Close(); err != nil {
					return err.Error()
				}
			}
			if d != nil {
				if err := d.Close(); err != nil {
					return err.Error()
				}
			}

			opts := &Options{}
			opts.BlockPropertyCollectors = []func() BlockPropertyCollector{
				func() BlockPropertyCollector {
					return &minSeqNumPropertyCollector{}
				},
			}

			var err error
			if d, err = runDBDefineCmd(td, opts); err != nil {
				return err.Error()
			}

			d.mu.Lock()
			// Disable the "dynamic base level" code for this test.
			// d.mu.versions.picker.forceBaseLevel1()
			s := d.mu.versions.currentVersion().String()
			d.mu.Unlock()
			return s

		case "set":
			if d == nil {
				return fmt.Sprintf("%s: db is not defined", td.Cmd)
			}

			var allowedSeeks int64
			td.ScanArgs(t, "allowed-seeks", &allowedSeeks)

			d.mu.Lock()
			for _, l := range d.mu.versions.currentVersion().Levels {
				for f := range l.All() {
					f.AllowedSeeks.Store(allowedSeeks)
				}
			}
			d.mu.Unlock()
			return ""

		case "show":
			if d == nil {
				return fmt.Sprintf("%s: db is not defined", td.Cmd)
			}

			var tableNum int64
			for _, arg := range td.CmdArgs {
				if len(arg.Vals) != 2 {
					return fmt.Sprintf("%s: %s=<value>", td.Cmd, arg.Key)
				}
				switch arg.Key {
				case "allowed-seeks":
					var err error
					tableNum, err = strconv.ParseInt(arg.Vals[0], 10, 64)
					if err != nil {
						return err.Error()
					}
				}
			}

			var foundAllowedSeeks int64 = -1
			d.mu.Lock()
			for _, l := range d.mu.versions.currentVersion().Levels {
				for f := range l.All() {
					if f.TableNum == base.TableNum(tableNum) {
						actualAllowedSeeks := f.AllowedSeeks.Load()
						foundAllowedSeeks = actualAllowedSeeks
					}
				}
			}
			d.mu.Unlock()

			if foundAllowedSeeks == -1 {
				return fmt.Sprintf("invalid file num: %d", tableNum)
			}
			return fmt.Sprintf("%d", foundAllowedSeeks)

		case "iter":
			if iter == nil || iter.iter == nil {
				// TODO(peter): runDBDefineCmd doesn't properly update the visible
				// sequence number. So we have to use a snapshot with a very large
				// sequence number, otherwise the DB appears empty.
				snap := Snapshot{
					db:     d,
					seqNum: base.SeqNumMax,
				}
				iter, _ = snap.NewIter(nil)
				iter.readSampling.forceReadSampling = true
			}
			return runIterCmd(td, iter, false)

		case "read-compactions":
			if d == nil {
				return fmt.Sprintf("%s: db is not defined", td.Cmd)
			}

			d.mu.Lock()
			var sb strings.Builder
			if d.mu.compact.readCompactions.size == 0 {
				sb.WriteString("(none)")
			}
			for i := 0; i < d.mu.compact.readCompactions.size; i++ {
				rc := d.mu.compact.readCompactions.at(i)
				sb.WriteString(fmt.Sprintf("(level: %d, start: %s, end: %s)\n", rc.level, string(rc.start), string(rc.end)))
			}
			d.mu.Unlock()
			return sb.String()

		case "iter-read-compactions":
			if iter == nil {
				return fmt.Sprintf("%s: iter is not defined", td.Cmd)
			}

			var sb strings.Builder
			if iter.readSampling.pendingCompactions.size == 0 {
				sb.WriteString("(none)")
			}
			for i := 0; i < iter.readSampling.pendingCompactions.size; i++ {
				rc := iter.readSampling.pendingCompactions.at(i)
				sb.WriteString(fmt.Sprintf("(level: %d, start: %s, end: %s)\n", rc.level, string(rc.start), string(rc.end)))
			}
			return sb.String()

		case "close-iter":
			if iter != nil {
				if err := iter.Close(); err != nil {
					return err.Error()
				}
			}
			return ""

		default:
			return fmt.Sprintf("unknown command: %s", td.Cmd)
		}
	})
}

func TestIteratorTableFilter(t *testing.T) {
	var d *DB
	defer func() {
		if d != nil {
			require.NoError(t, d.Close())
		}
	}()

	datadriven.RunTest(t, "testdata/iterator_table_filter", func(t *testing.T, td *datadriven.TestData) string {
		switch td.Cmd {
		case "define":
			if d != nil {
				if err := d.Close(); err != nil {
					return err.Error()
				}
			}

			opts := &Options{}
			opts.BlockPropertyCollectors = []func() BlockPropertyCollector{
				func() BlockPropertyCollector {
					return &minSeqNumPropertyCollector{}
				},
			}

			var err error
			if d, err = runDBDefineCmd(td, opts); err != nil {
				return err.Error()
			}

			d.mu.Lock()
			// Disable the "dynamic base level" code for this test.
			d.mu.versions.picker.forceBaseLevel1()
			s := d.mu.versions.currentVersion().String()
			d.mu.Unlock()
			return s

		case "iter":
			// We're using an iterator table filter to approximate what is done by
			// snapshots.
			iterOpts := &IterOptions{}
			var filterSeqNum uint64
			if td.MaybeScanArgs(t, "filter", &filterSeqNum) {
				iterOpts.PointKeyFilters = []BlockPropertyFilter{
					&minSeqNumFilter{seqNumUpperBound: filterSeqNum},
				}
			}

			// TODO(peter): runDBDefineCmd doesn't properly update the visible
			// sequence number. So we have to use a snapshot with a very large
			// sequence number, otherwise the DB appears empty.
			snap := Snapshot{
				db:     d,
				seqNum: base.SeqNumMax,
			}
			iter, _ := snap.NewIter(iterOpts)
			return runIterCmd(td, iter, true)

		default:
			return fmt.Sprintf("unknown command: %s", td.Cmd)
		}
	})
}

func TestIteratorNextPrev(t *testing.T) {
	var mem vfs.FS
	var d *DB
	defer func() {
		require.NoError(t, d.Close())
	}()

	reset := func() {
		if d != nil {
			require.NoError(t, d.Close())
		}

		mem = vfs.NewMem()
		require.NoError(t, mem.MkdirAll("ext", 0755))
		opts := &Options{FS: mem}
		// Automatic compactions may compact away tombstones from L6, making
		// some testcases non-deterministic.
		opts.DisableAutomaticCompactions = true
		var err error
		d, err = Open("", opts)
		require.NoError(t, err)
	}
	reset()

	datadriven.RunTest(t, "testdata/iterator_next_prev", func(t *testing.T, td *datadriven.TestData) string {
		switch td.Cmd {
		case "reset":
			reset()
			return ""

		case "build":
			if err := runBuildCmd(td, d, mem); err != nil {
				return err.Error()
			}
			return ""

		case "ingest":
			if err := runIngestCmd(td, d, mem); err != nil {
				return err.Error()
			}
			return runLSMCmd(td, d)

		case "iter":
			snap := Snapshot{
				db:     d,
				seqNum: base.SeqNumMax,
			}
			if td.HasArg("seq") {
				var n uint64
				td.ScanArgs(t, "seq", &n)
				snap.seqNum = base.SeqNum(n)
			}
			iter, _ := snap.NewIter(nil)
			return runIterCmd(td, iter, true)

		default:
			return fmt.Sprintf("unknown command: %s", td.Cmd)
		}
	})
}

func TestIteratorStats(t *testing.T) {
	var mem vfs.FS
	var d *DB
	defer func() {
		require.NoError(t, d.Close())
	}()

	reset := func() {
		if d != nil {
			require.NoError(t, d.Close())
		}

		mem = vfs.NewMem()
		require.NoError(t, mem.MkdirAll("ext", 0755))
		opts := &Options{Comparer: testkeys.Comparer, FS: mem, FormatMajorVersion: internalFormatNewest}
		// Automatic compactions may make some testcases non-deterministic.
		opts.DisableAutomaticCompactions = true
		opts.Experimental.EnableColumnarBlocks = func() bool { return true }
		var err error
		d, err = Open("", opts)
		require.NoError(t, err)
	}
	reset()

	datadriven.RunTest(t, "testdata/iterator_stats", func(t *testing.T, td *datadriven.TestData) string {
		switch td.Cmd {
		case "reset":
			reset()
			return ""

		case "build":
			if err := runBuildCmd(td, d, mem); err != nil {
				return err.Error()
			}
			return ""

		case "ingest":
			if err := runIngestCmd(td, d, mem); err != nil {
				return err.Error()
			}
			return runLSMCmd(td, d)

		case "iter":
			snap := Snapshot{
				db:     d,
				seqNum: base.SeqNumMax,
			}
			td.MaybeScanArgs(t, "seq", &snap.seqNum)
			iter, _ := snap.NewIter(nil)
			return runIterCmd(td, iter, true)

		default:
			return fmt.Sprintf("unknown command: %s", td.Cmd)
		}
	})
}

type iterSeekOptWrapper struct {
	internalIterator

	seekGEUsingNext, seekPrefixGEUsingNext *int
}

func (i *iterSeekOptWrapper) SeekGE(key []byte, flags base.SeekGEFlags) *base.InternalKV {
	if flags.TrySeekUsingNext() {
		*i.seekGEUsingNext++
	}
	return i.internalIterator.SeekGE(key, flags)
}

func (i *iterSeekOptWrapper) SeekPrefixGE(
	prefix, key []byte, flags base.SeekGEFlags,
) *base.InternalKV {
	if flags.TrySeekUsingNext() {
		*i.seekPrefixGEUsingNext++
	}
	return i.internalIterator.SeekPrefixGE(prefix, key, flags)
}

func TestIteratorSeekOpt(t *testing.T) {
	var d *DB
	defer func() {
		require.NoError(t, d.Close())
	}()
	var iter *Iterator
	defer func() {
		if iter != nil {
			require.NoError(t, iter.Close())
		}
	}()
	var seekGEUsingNext, seekPrefixGEUsingNext int

	datadriven.RunTest(t, "testdata/iterator_seek_opt", func(t *testing.T, td *datadriven.TestData) string {
		switch td.Cmd {
		case "define":
			if iter != nil {
				if err := iter.Close(); err != nil {
					return err.Error()
				}
			}
			if d != nil {
				if err := d.Close(); err != nil {
					return err.Error()
				}
			}
			seekGEUsingNext = 0
			seekPrefixGEUsingNext = 0

			opts := &Options{}
			opts.BlockPropertyCollectors = []func() BlockPropertyCollector{
				func() BlockPropertyCollector {
					return &minSeqNumPropertyCollector{}
				},
			}

			var err error
			if d, err = runDBDefineCmd(td, opts); err != nil {
				return err.Error()
			}

			d.mu.Lock()
			s := d.mu.versions.currentVersion().String()
			d.mu.Unlock()
			oldNewIters := d.newIters
			d.newIters = func(
				ctx context.Context, file *manifest.TableMetadata, opts *IterOptions,
				internalOpts internalIterOpts, kinds iterKinds) (iterSet, error) {
				iters, err := oldNewIters(ctx, file, opts, internalOpts, kinds)
				iters.point = &iterSeekOptWrapper{
					internalIterator:      iters.point,
					seekGEUsingNext:       &seekGEUsingNext,
					seekPrefixGEUsingNext: &seekPrefixGEUsingNext,
				}
				return iters, err
			}
			d.opts.Comparer.Split = func(a []byte) int { return len(a) }
			return s

		case "iter":
			if iter == nil || iter.iter == nil {
				// TODO(peter): runDBDefineCmd doesn't properly update the visible
				// sequence number. So we have to use a snapshot with a very large
				// sequence number, otherwise the DB appears empty.
				snap := Snapshot{
					db:     d,
					seqNum: base.SeqNumMax,
				}
				iter, _ = snap.NewIter(nil)
				iter.readSampling.forceReadSampling = true
				iter.comparer.Split = d.opts.Comparer.Split
				iter.forceEnableSeekOpt = true
				iter.merging.forceEnableSeekOpt = true
			}
			iterOutput := runIterCmd(td, iter, false)
			stats := iter.Stats()
			// InternalStats are non-deterministic since they depend on how data is
			// distributed across memtables and sstables in the DB.
			stats.InternalStats = InternalIteratorStats{}
			var builder strings.Builder
			fmt.Fprintf(&builder, "%sstats: %s\n", iterOutput, stats.String())
			fmt.Fprintf(&builder, "SeekGEs with trySeekUsingNext: %d\n", seekGEUsingNext)
			fmt.Fprintf(&builder, "SeekPrefixGEs with trySeekUsingNext: %d\n", seekPrefixGEUsingNext)
			return builder.String()

		default:
			return fmt.Sprintf("unknown command: %s", td.Cmd)
		}
	})
}

type errorSeekIter struct {
	internalIterator
	// Fields controlling error injection for seeks.
	injectSeekErrorCounts []int
	seekCount             int
	err                   error
}

func (i *errorSeekIter) SeekGE(key []byte, flags base.SeekGEFlags) *base.InternalKV {
	if i.tryInjectError() {
		return nil
	}
	i.err = nil
	i.seekCount++
	return i.internalIterator.SeekGE(key, flags)
}

func (i *errorSeekIter) SeekPrefixGE(prefix, key []byte, flags base.SeekGEFlags) *base.InternalKV {
	if i.tryInjectError() {
		return nil
	}
	i.err = nil
	i.seekCount++
	return i.internalIterator.SeekPrefixGE(prefix, key, flags)
}

func (i *errorSeekIter) SeekLT(key []byte, flags base.SeekLTFlags) *base.InternalKV {
	if i.tryInjectError() {
		return nil
	}
	i.err = nil
	i.seekCount++
	return i.internalIterator.SeekLT(key, flags)
}

func (i *errorSeekIter) tryInjectError() bool {
	if len(i.injectSeekErrorCounts) > 0 && i.injectSeekErrorCounts[0] == i.seekCount {
		i.seekCount++
		i.err = errors.Errorf("injecting error")
		i.injectSeekErrorCounts = i.injectSeekErrorCounts[1:]
		return true
	}
	return false
}

func (i *errorSeekIter) First() *base.InternalKV {
	i.err = nil
	return i.internalIterator.First()
}

func (i *errorSeekIter) Last() *base.InternalKV {
	i.err = nil
	return i.internalIterator.Last()
}

func (i *errorSeekIter) Next() *base.InternalKV {
	if i.err != nil {
		return nil
	}
	return i.internalIterator.Next()
}

func (i *errorSeekIter) Prev() *base.InternalKV {
	if i.err != nil {
		return nil
	}
	return i.internalIterator.Prev()
}

func (i *errorSeekIter) Error() error {
	if i.err != nil {
		return i.err
	}
	return i.internalIterator.Error()
}

func TestIteratorSeekOptErrors(t *testing.T) {
	var kvs []base.InternalKV

	var errorIter errorSeekIter
	newIter := func(opts IterOptions) *Iterator {
		iter := base.NewFakeIter(kvs)
		iter.SetBounds(opts.GetLowerBound(), opts.GetUpperBound())
		errorIter = errorSeekIter{internalIterator: invalidating.NewIter(iter)}
		// NB: This Iterator cannot be cloned since it is not constructed
		// with a readState. It suffices for this test.
		return &Iterator{
			opts:     opts,
			comparer: *testkeys.Comparer,
			merge:    DefaultMerger.Merge,
			iter:     &errorIter,
		}
	}

	datadriven.RunTest(t, "testdata/iterator_seek_opt_errors", func(t *testing.T, d *datadriven.TestData) string {
		switch d.Cmd {
		case "define":
			kvs = kvs[:0]
			for _, key := range strings.Split(d.Input, "\n") {
				j := strings.Index(key, ":")
				kvs = append(kvs, base.MakeInternalKV(base.ParseInternalKey(key[:j]), []byte(key[j+1:])))
			}
			return ""

		case "iter":
			var opts IterOptions
			var injectSeekGEErrorCounts []int
			for _, arg := range d.CmdArgs {
				if len(arg.Vals) < 1 {
					return fmt.Sprintf("%s: %s=<value>", d.Cmd, arg.Key)
				}
				switch arg.Key {
				case "lower":
					opts.LowerBound = []byte(arg.Vals[0])
				case "upper":
					opts.UpperBound = []byte(arg.Vals[0])
				case "seek-error":
					for i := 0; i < len(arg.Vals); i++ {
						n, err := strconv.Atoi(arg.Vals[i])
						if err != nil {
							return err.Error()
						}
						injectSeekGEErrorCounts = append(injectSeekGEErrorCounts, n)
					}
				default:
					return fmt.Sprintf("%s: unknown arg: %s", d.Cmd, arg.Key)
				}
			}

			iter := newIter(opts)
			errorIter.injectSeekErrorCounts = injectSeekGEErrorCounts
			return runIterCmd(d, iter, true)

		default:
			return fmt.Sprintf("unknown command: %s", d.Cmd)
		}
	})
}

type testBlockIntervalMapper struct {
	numLength     int
	offsetFromEnd int
}

var _ sstable.IntervalMapper = (*testBlockIntervalMapper)(nil)

func (bi *testBlockIntervalMapper) MapPointKey(
	key InternalKey, value []byte,
) (sstable.BlockInterval, error) {
	k := key.UserKey
	if len(k) < bi.numLength+bi.offsetFromEnd {
		return sstable.BlockInterval{}, nil
	}
	n := len(k) - bi.offsetFromEnd - bi.numLength
	val, err := strconv.Atoi(string(k[n : n+bi.numLength]))
	if err != nil {
		return sstable.BlockInterval{}, err
	}
	if val < 0 {
		panic("testBlockIntervalMapper expects values >= 0")
	}
	uval := uint64(val)
	return sstable.BlockInterval{Lower: uval, Upper: uval + 1}, nil
}

func (bi *testBlockIntervalMapper) MapRangeKeys(span sstable.Span) (sstable.BlockInterval, error) {
	return sstable.BlockInterval{}, nil
}
func TestIteratorBlockIntervalFilter(t *testing.T) {
	var mem vfs.FS
	var d *DB
	defer func() {
		require.NoError(t, d.Close())
	}()

	type collector struct {
		id     uint16
		offset int
	}
	createDB := func(collectors []collector) {
		if d != nil {
			require.NoError(t, d.Close())
		}

		mem = vfs.NewMem()
		require.NoError(t, mem.MkdirAll("ext", 0755))

		var bpCollectors []func() BlockPropertyCollector
		for _, c := range collectors {
			coll := c
			bpCollectors = append(bpCollectors, func() BlockPropertyCollector {
				return sstable.NewBlockIntervalCollector(
					fmt.Sprintf("%d", coll.id),
					&testBlockIntervalMapper{numLength: 2, offsetFromEnd: coll.offset},
					nil, /* range key collector */
				)
			})
		}
		opts := &Options{
			FS:                      mem,
			FormatMajorVersion:      internalFormatNewest,
			BlockPropertyCollectors: bpCollectors,
		}
		opts.Levels[0] = LevelOptions{BlockSize: 1, IndexBlockSize: 1}

		// Automatic compactions may compact away tombstones from L6, making
		// some testcases non-deterministic.
		opts.DisableAutomaticCompactions = true
		var err error
		d, err = Open("", opts)
		require.NoError(t, err)
	}

	datadriven.RunTest(
		t, "testdata/iterator_block_interval_filter", func(t *testing.T, td *datadriven.TestData) string {
			switch td.Cmd {
			case "build":
				var collectors []collector
				for _, arg := range td.CmdArgs {
					switch arg.Key {
					case "id_offset":
						if len(arg.Vals) != 2 {
							return "id and offset not provided"
						}
						var id, offset int
						var err error
						if id, err = strconv.Atoi(arg.Vals[0]); err != nil {
							return err.Error()
						}
						if offset, err = strconv.Atoi(arg.Vals[1]); err != nil {
							return err.Error()
						}
						collectors = append(collectors, collector{id: uint16(id), offset: offset})
					default:
						return fmt.Sprintf("unknown key: %s", arg.Key)
					}
				}
				createDB(collectors)
				b := d.NewBatch()
				if err := runBatchDefineCmd(td, b); err != nil {
					return err.Error()
				}
				if err := b.Commit(nil); err != nil {
					return err.Error()
				}
				if err := d.Flush(); err != nil {
					return err.Error()
				}
				return runLSMCmd(td, d)

			case "iter":
				var opts IterOptions
				for _, arg := range td.CmdArgs {
					switch arg.Key {
					case "id_lower_upper":
						if len(arg.Vals) != 3 {
							return "id, lower, upper not provided"
						}
						var id, lower, upper int
						var err error
						if id, err = strconv.Atoi(arg.Vals[0]); err != nil {
							return err.Error()
						}
						if lower, err = strconv.Atoi(arg.Vals[1]); err != nil {
							return err.Error()
						}
						if upper, err = strconv.Atoi(arg.Vals[2]); err != nil {
							return err.Error()
						}
						opts.PointKeyFilters = append(opts.PointKeyFilters,
							sstable.NewBlockIntervalFilter(fmt.Sprintf("%d", id), uint64(lower), uint64(upper), nil))
					default:
						return fmt.Sprintf("unknown key: %s", arg.Key)
					}
				}
				rand.Shuffle(len(opts.PointKeyFilters), func(i, j int) {
					opts.PointKeyFilters[i], opts.PointKeyFilters[j] =
						opts.PointKeyFilters[j], opts.PointKeyFilters[i]
				})
				iter, _ := d.NewIter(&opts)
				return runIterCmd(td, iter, true)

			default:
				return fmt.Sprintf("unknown command: %s", td.Cmd)
			}
		})
}

var seed = flag.Uint64("seed", 0, "a pseudorandom number generator seed")

func randStr(fill []byte, rng *rand.Rand) {
	const letters = "abcdefghijklmnopqrstuvwxyz"
	const lettersLen = len(letters)
	for i := 0; i < len(fill); i++ {
		fill[i] = letters[rng.IntN(lettersLen)]
	}
}

func randValue(n int, rng *rand.Rand) []byte {
	buf := make([]byte, n)
	randStr(buf, rng)
	return buf
}

func randKey(n int, rng *rand.Rand) ([]byte, int) {
	keyPrefix := randValue(n, rng)
	suffix := rng.IntN(100)
	return append(keyPrefix, []byte(fmt.Sprintf("%02d", suffix))...), suffix
}

func TestIteratorRandomizedBlockIntervalFilter(t *testing.T) {
	mem := vfs.NewMem()
	opts := &Options{
		FS:                 mem,
		FormatMajorVersion: internalFormatNewest,
		BlockPropertyCollectors: []func() BlockPropertyCollector{
			func() BlockPropertyCollector {
				return sstable.NewBlockIntervalCollector(
					"0", &testBlockIntervalMapper{numLength: 2}, nil, /* suffixReplacer */
				)
			},
		},
	}
	seed := *seed
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
		t.Logf("seed: %d", seed)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	opts.FlushSplitBytes = 1 << rng.IntN(8)            // 1B - 256B
	opts.L0CompactionThreshold = 1 << rng.IntN(2)      // 1-2
	opts.L0CompactionFileThreshold = 1 << rng.IntN(11) // 1-1024
	opts.LBaseMaxBytes = 1 << rng.IntN(11)             // 1B - 1KB
	opts.MemTableSize = 2 << 10                        // 2KB
	opts.Levels[0].BlockSize = 1 << rng.IntN(8)        // 1B - 256B
	opts.Levels[0].IndexBlockSize = 1 << rng.IntN(8)   // 1B - 256B

	d, err := Open("", opts)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, d.Close())
	}()
	matchingKeyValues := make(map[string]string)
	lower := rng.IntN(100)
	upper := rng.IntN(100)
	if lower > upper {
		lower, upper = upper, lower
	}
	n := 2000
	for i := 0; i < n; i++ {
		key, suffix := randKey(20+rng.IntN(5), rng)
		value := randValue(50, rng)
		if lower <= suffix && suffix < upper {
			matchingKeyValues[string(key)] = string(value)
		}
		d.Set(key, value, nil)
	}

	var iterOpts IterOptions
	iterOpts.PointKeyFilters = []BlockPropertyFilter{
		sstable.NewBlockIntervalFilter("0", uint64(lower), uint64(upper), nil),
	}
	iter, _ := d.NewIter(&iterOpts)
	defer func() {
		require.NoError(t, iter.Close())
	}()
	iter.First()
	found := 0
	matchingCount := len(matchingKeyValues)
	for ; iter.Valid(); iter.Next() {
		found++
		key := string(iter.Key())
		value, ok := matchingKeyValues[key]
		if ok {
			require.Equal(t, value, string(iter.Value()))
			delete(matchingKeyValues, key)
		}
	}
	t.Logf("generated %d keys: %d matching, %d found", n, matchingCount, found)
	require.Equal(t, 0, len(matchingKeyValues))
}

func TestIteratorGuaranteedDurable(t *testing.T) {
	mem := vfs.NewMem()
	opts := &Options{FS: mem}
	d, err := Open("", opts)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, d.Close())
	}()
	iterOptions := IterOptions{OnlyReadGuaranteedDurable: true}
	failFunc := func(t *testing.T, reader Reader) {
		defer func() {
			if r := recover(); r == nil {
				require.Fail(t, "expected panic")
			}
			reader.Close()
		}()
		iter, _ := reader.NewIter(&iterOptions)
		defer iter.Close()
	}
	t.Run("snapshot", func(t *testing.T) {
		failFunc(t, d.NewSnapshot())
	})
	t.Run("batch", func(t *testing.T) {
		failFunc(t, d.NewIndexedBatch())
	})
	t.Run("db", func(t *testing.T) {
		d.Set([]byte("k"), []byte("v"), nil)
		foundKV := func(o *IterOptions) bool {
			iter, _ := d.NewIter(o)
			defer iter.Close()
			iter.SeekGE([]byte("k"))
			return iter.Valid()
		}
		require.True(t, foundKV(nil))
		require.False(t, foundKV(&iterOptions))
		require.NoError(t, d.Flush())
		require.True(t, foundKV(nil))
		require.True(t, foundKV(&iterOptions))
	})
}

func TestIteratorBoundsLifetimes(t *testing.T) {
	rng := rand.New(rand.NewPCG(0, uint64(time.Now().UnixNano())))
	d := newPointTestkeysDatabase(t, testkeys.Alpha(2))
	defer func() { require.NoError(t, d.Close()) }()

	var buf bytes.Buffer
	iterators := map[string]*Iterator{}
	var labels []string
	printIters := func(w io.Writer) {
		labels = labels[:0]
		for label := range iterators {
			labels = append(labels, label)
		}
		sort.Strings(labels)
		for _, label := range labels {
			it := iterators[label]
			fmt.Fprintf(&buf, "%s: (", label)
			if it.opts.LowerBound == nil {
				fmt.Fprint(&buf, "<nil>, ")
			} else {
				fmt.Fprintf(&buf, "%q, ", it.opts.LowerBound)
			}
			if it.opts.UpperBound == nil {
				fmt.Fprint(&buf, "<nil>)")
			} else {
				fmt.Fprintf(&buf, "%q)", it.opts.UpperBound)
			}
			fmt.Fprintf(&buf, " boundsBufIdx=%d\n", it.boundsBufIdx)
		}
	}
	parseBounds := func(td *datadriven.TestData) (lower, upper []byte) {
		for _, arg := range td.CmdArgs {
			if arg.Key == "lower" {
				lower = []byte(arg.Vals[0])
			} else if arg.Key == "upper" {
				upper = []byte(arg.Vals[0])
			}
		}
		return lower, upper
	}
	trashBounds := func(bounds ...[]byte) {
		for _, bound := range bounds {
			for j := range bound {
				bound[j] = byte(rng.Uint32())
			}
		}
	}

	datadriven.RunTest(t, "testdata/iterator_bounds_lifetimes", func(t *testing.T, td *datadriven.TestData) string {
		switch td.Cmd {
		case "define":
			var err error
			if d, err = runDBDefineCmd(td, d.opts); err != nil {
				return err.Error()
			}
			d.mu.Lock()
			s := d.mu.versions.currentVersion().String()
			d.mu.Unlock()
			return s
		case "new-iter":
			var label string
			td.ScanArgs(t, "label", &label)
			lower, upper := parseBounds(td)
			iterators[label], _ = d.NewIter(&IterOptions{
				LowerBound: lower,
				UpperBound: upper,
			})
			trashBounds(lower, upper)
			buf.Reset()
			printIters(&buf)
			return buf.String()
		case "clone":
			var from, to string
			td.ScanArgs(t, "from", &from)
			td.ScanArgs(t, "to", &to)
			var err error
			iterators[to], err = iterators[from].Clone(CloneOptions{})
			if err != nil {
				return err.Error()
			}
			buf.Reset()
			printIters(&buf)
			return buf.String()
		case "close":
			var label string
			td.ScanArgs(t, "label", &label)
			iterators[label].Close()
			delete(iterators, label)
			buf.Reset()
			printIters(&buf)
			return buf.String()
		case "iter":
			var label string
			td.ScanArgs(t, "label", &label)
			return runIterCmd(td, iterators[label], false /* closeIter */)
		case "set-bounds":
			var label string
			td.ScanArgs(t, "label", &label)
			lower, upper := parseBounds(td)
			iterators[label].SetBounds(lower, upper)
			trashBounds(lower, upper)
			buf.Reset()
			printIters(&buf)
			return buf.String()
		case "set-options":
			var label string
			td.ScanArgs(t, "label", &label)
			opts := iterators[label].opts
			for _, arg := range td.CmdArgs {
				if arg.Key == "key-types" {
					switch arg.Vals[0] {
					case "points-only":
						opts.KeyTypes = IterKeyTypePointsOnly
					case "ranges-only":
						opts.KeyTypes = IterKeyTypeRangesOnly
					case "both":
						opts.KeyTypes = IterKeyTypePointsAndRanges
					default:
						panic(fmt.Sprintf("unrecognized key type %q", arg.Vals[0]))
					}
				}
			}
			opts.LowerBound, opts.UpperBound = parseBounds(td)
			iterators[label].SetOptions(&opts)
			trashBounds(opts.LowerBound, opts.UpperBound)
			buf.Reset()
			printIters(&buf)
			return buf.String()
		default:
			return fmt.Sprintf("unrecognized command %q", td.Cmd)
		}
	})
}

func TestIteratorStatsMerge(t *testing.T) {
	s := IteratorStats{
		ForwardSeekCount: [NumStatsKind]int{1, 2},
		ReverseSeekCount: [NumStatsKind]int{3, 4},
		ForwardStepCount: [NumStatsKind]int{5, 6},
		ReverseStepCount: [NumStatsKind]int{7, 8},
		InternalStats: InternalIteratorStats{
			BlockReads: [blockkind.NumKinds]base.BlockReadStats{
				blockkind.SSTableData: {
					Count:             2,
					CountInCache:      1,
					BlockBytes:        9,
					BlockBytesInCache: 10,
					BlockReadDuration: 3 * time.Millisecond,
				},
			},
			KeyBytes:                       11,
			ValueBytes:                     12,
			PointCount:                     13,
			PointsCoveredByRangeTombstones: 14,
		},
		RangeKeyStats: RangeKeyIteratorStats{
			Count:           15,
			ContainedPoints: 16,
			SkippedPoints:   17,
		},
	}
	s.InternalStats.SeparatedPointValue.Count = 1
	s.InternalStats.SeparatedPointValue.ValueBytes = 5
	s.InternalStats.SeparatedPointValue.ValueBytesFetched = 3
	s2 := IteratorStats{
		ForwardSeekCount: [NumStatsKind]int{1, 2},
		ReverseSeekCount: [NumStatsKind]int{3, 4},
		ForwardStepCount: [NumStatsKind]int{5, 6},
		ReverseStepCount: [NumStatsKind]int{7, 8},
		InternalStats: InternalIteratorStats{
			BlockReads: [blockkind.NumKinds]base.BlockReadStats{
				blockkind.SSTableData: {
					Count:             1,
					CountInCache:      1,
					BlockBytes:        9,
					BlockBytesInCache: 10,
					BlockReadDuration: 3 * time.Millisecond,
				},
			},
			KeyBytes:                       11,
			ValueBytes:                     12,
			PointCount:                     13,
			PointsCoveredByRangeTombstones: 14,
		},
		RangeKeyStats: RangeKeyIteratorStats{
			Count:           15,
			ContainedPoints: 16,
			SkippedPoints:   17,
		},
	}
	s2.InternalStats.SeparatedPointValue.Count = 2
	s2.InternalStats.SeparatedPointValue.ValueBytes = 10
	s2.InternalStats.SeparatedPointValue.ValueBytesFetched = 6
	s.Merge(s2)
	expected := IteratorStats{
		ForwardSeekCount: [NumStatsKind]int{2, 4},
		ReverseSeekCount: [NumStatsKind]int{6, 8},
		ForwardStepCount: [NumStatsKind]int{10, 12},
		ReverseStepCount: [NumStatsKind]int{14, 16},
		InternalStats: InternalIteratorStats{
			BlockReads: [blockkind.NumKinds]base.BlockReadStats{
				blockkind.SSTableData: {
					Count:             3,
					CountInCache:      2,
					BlockBytes:        18,
					BlockBytesInCache: 20,
					BlockReadDuration: 6 * time.Millisecond,
				},
			},
			KeyBytes:                       22,
			ValueBytes:                     24,
			PointCount:                     26,
			PointsCoveredByRangeTombstones: 28,
		},
		RangeKeyStats: RangeKeyIteratorStats{
			Count:           30,
			ContainedPoints: 32,
			SkippedPoints:   34,
		},
	}
	expected.InternalStats.SeparatedPointValue.Count = 3
	expected.InternalStats.SeparatedPointValue.ValueBytes = 15
	expected.InternalStats.SeparatedPointValue.ValueBytesFetched = 9
	require.Equal(t, expected, s)
}

// TestSetOptionsEquivalence tests equivalence between SetOptions to mutate an
// iterator and constructing a new iterator with NewIter. The long-lived
// iterator and the new iterator should surface identical iterator states.
func TestSetOptionsEquivalence(t *testing.T) {
	seed := uint64(time.Now().UnixNano())
	// Call a helper function with the seed so that the seed appears within
	// stack traces if there's a panic.
	testSetOptionsEquivalence(t, seed)
}

func testSetOptionsEquivalence(t *testing.T, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, seed))
	ks := testkeys.Alpha(2)
	d := newTestkeysDatabase(t, ks, rng)
	defer func() { require.NoError(t, d.Close()) }()

	var o IterOptions
	generateNewOptions := func() {
		// TODO(jackson): Include test coverage for block property filters, etc.
		if rng.IntN(2) == 1 {
			o.KeyTypes = IterKeyType(rng.IntN(3))
		}
		if rng.IntN(2) == 1 {
			if rng.IntN(2) == 1 {
				o.LowerBound = nil
				if rng.IntN(2) == 1 {
					o.LowerBound = testkeys.KeyAt(ks, rng.Int64N(ks.Count()), rng.Int64N(ks.Count()))
				}
			}
			if rng.IntN(2) == 1 {
				o.UpperBound = nil
				if rng.IntN(2) == 1 {
					o.UpperBound = testkeys.KeyAt(ks, rng.Int64N(ks.Count()), rng.Int64N(ks.Count()))
				}
			}
			if testkeys.Comparer.Compare(o.LowerBound, o.UpperBound) > 0 {
				o.LowerBound, o.UpperBound = o.UpperBound, o.LowerBound
			}
		}
		o.RangeKeyMasking.Suffix = nil
		if o.KeyTypes == IterKeyTypePointsAndRanges && rng.IntN(2) == 1 {
			o.RangeKeyMasking.Suffix = testkeys.Suffix(rng.Int64N(ks.Count()))
		}
	}

	var longLivedIter, newIter *Iterator
	var history, longLivedBuf, newIterBuf bytes.Buffer
	defer func() {
		if r := recover(); r != nil {
			t.Log(history.String())
			panic(r)
		}
	}()
	defer func() {
		if longLivedIter != nil {
			longLivedIter.Close()
		}
		if newIter != nil {
			newIter.Close()
		}
	}()

	type positioningOp struct {
		desc string
		run  func(*Iterator) IterValidityState
	}
	positioningOps := []func() positioningOp{
		// SeekGE
		func() positioningOp {
			k := testkeys.Key(ks, rng.Int64N(ks.Count()))
			return positioningOp{
				desc: fmt.Sprintf("SeekGE(%q)", k),
				run: func(it *Iterator) IterValidityState {
					return it.SeekGEWithLimit(k, nil)
				},
			}
		},
		// SeekLT
		func() positioningOp {
			k := testkeys.Key(ks, rng.Int64N(ks.Count()))
			return positioningOp{
				desc: fmt.Sprintf("SeekLT(%q)", k),
				run: func(it *Iterator) IterValidityState {
					return it.SeekLTWithLimit(k, nil)
				},
			}
		},
		// SeekPrefixGE
		func() positioningOp {
			k := testkeys.Key(ks, rng.Int64N(ks.Count()))
			return positioningOp{
				desc: fmt.Sprintf("SeekPrefixGE(%q)", k),
				run: func(it *Iterator) IterValidityState {
					if it.SeekPrefixGE(k) {
						return IterValid
					}
					return IterExhausted
				},
			}
		},
	}

	for i := 0; i < 10_000; i++ {
		// Generate new random options. The options in o will be mutated.
		generateNewOptions()
		fmt.Fprintf(&history, "new options: %s\n", iterOptionsString(&o))

		newIter, _ = d.NewIter(&o)
		if longLivedIter == nil {
			longLivedIter, _ = d.NewIter(&o)
		} else {
			longLivedIter.SetOptions(&o)
		}

		// Apply the same operation to both keys.
		iterOp := positioningOps[rng.IntN(len(positioningOps))]()
		newIterValidity := iterOp.run(newIter)
		longLivedValidity := iterOp.run(longLivedIter)

		newIterBuf.Reset()
		longLivedBuf.Reset()
		printIterState(&newIterBuf, newIter, newIterValidity, true /* printValidityState */)
		printIterState(&longLivedBuf, longLivedIter, longLivedValidity, true /* printValidityState */)
		fmt.Fprintf(&history, "%s = %s\n", iterOp.desc, newIterBuf.String())

		if newIterBuf.String() != longLivedBuf.String() {
			t.Logf("history:\n%s\n", history.String())
			t.Logf("seed: %d\n", seed)
			t.Fatalf("expected %q, got %q", newIterBuf.String(), longLivedBuf.String())
		}
		_ = newIter.Close()

		newIter = nil
	}
	t.Logf("history:\n%s\n", history.String())
}

func iterOptionsString(o *IterOptions) string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "key-types=%s, lower=%q, upper=%q",
		o.KeyTypes, o.LowerBound, o.UpperBound)
	if o.OnlyReadGuaranteedDurable {
		fmt.Fprintf(&buf, ", only-durable")
	}
	if o.UseL6Filters {
		fmt.Fprintf(&buf, ", use-L6-filters")
	}
	for i, pkf := range o.PointKeyFilters {
		fmt.Fprintf(&buf, ", point-key-filter[%d]=%q", i, pkf.Name())
	}
	for i, rkf := range o.RangeKeyFilters {
		fmt.Fprintf(&buf, ", range-key-filter[%d]=%q", i, rkf.Name())
	}
	return buf.String()
}

func newTestkeysDatabase(t *testing.T, ks testkeys.Keyspace, rng *rand.Rand) *DB {
	dbOpts := &Options{
		Comparer: testkeys.Comparer,
		FS:       vfs.NewMem(),
		Logger:   panicLogger{},
	}
	dbOpts.testingRandomized(t)
	d, err := Open("", dbOpts)
	require.NoError(t, err)

	// Randomize the order in which we write keys.
	order := rng.Perm(int(ks.Count()))
	b := d.NewBatch()
	keyBuf := make([]byte, ks.MaxLen()+testkeys.MaxSuffixLen)
	keyBuf2 := make([]byte, ks.MaxLen()+testkeys.MaxSuffixLen)
	for i := 0; i < len(order); i++ {
		const maxVersionsPerKey = 10
		keyIndex := order[i]
		for versions := rng.IntN(maxVersionsPerKey); versions > 0; versions-- {
			n := testkeys.WriteKeyAt(keyBuf, ks, int64(keyIndex), rng.Int64N(maxVersionsPerKey))
			b.Set(keyBuf[:n], keyBuf[:n], nil)
		}

		// Sometimes add a range key too.
		if rng.IntN(100) == 1 {
			startIdx := rng.Int64N(ks.Count())
			endIdx := rng.Int64N(ks.Count())
			startLen := testkeys.WriteKey(keyBuf, ks, startIdx)
			endLen := testkeys.WriteKey(keyBuf2, ks, endIdx)
			suffixInt := rng.Int64N(maxVersionsPerKey)
			require.NoError(t, b.RangeKeySet(
				keyBuf[:startLen],
				keyBuf2[:endLen],
				testkeys.Suffix(suffixInt),
				nil,
				nil))
		}

		// Randomize the flush points.
		if !b.Empty() && rng.IntN(10) == 1 {
			require.NoError(t, b.Commit(nil))
			require.NoError(t, d.Flush())
			b = d.NewBatch()
		}
	}
	if !b.Empty() {
		require.NoError(t, b.Commit(nil))
	}
	return d
}

func newPointTestkeysDatabase(t *testing.T, ks testkeys.Keyspace) *DB {
	dbOpts := &Options{
		Comparer: testkeys.Comparer,
		FS:       vfs.NewMem(),
	}
	dbOpts.testingRandomized(t)
	d, err := Open("", dbOpts)
	require.NoError(t, err)

	b := d.NewBatch()
	keyBuf := make([]byte, ks.MaxLen()+testkeys.MaxSuffixLen)
	for i := int64(0); i < ks.Count(); i++ {
		n := testkeys.WriteKeyAt(keyBuf, ks, i, i)
		b.Set(keyBuf[:n], keyBuf[:n], nil)
	}
	require.NoError(t, b.Commit(nil))
	return d
}

func BenchmarkIteratorSeekGE(b *testing.B) {
	m, keys := buildMemTable(b)
	iter := &Iterator{
		comparer: *DefaultComparer,
		iter:     m.newIter(nil),
	}
	rng := rand.New(rand.NewPCG(0, uint64(time.Now().UnixNano())))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := keys[rng.IntN(len(keys))]
		iter.SeekGE(key)
	}
}

func BenchmarkIteratorNext(b *testing.B) {
	m, _ := buildMemTable(b)
	iter := &Iterator{
		comparer: *DefaultComparer,
		iter:     m.newIter(nil),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !iter.Valid() {
			iter.First()
		}
		iter.Next()
	}
}

func BenchmarkIteratorPrev(b *testing.B) {
	m, _ := buildMemTable(b)
	iter := &Iterator{
		comparer: *DefaultComparer,
		iter:     m.newIter(nil),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !iter.Valid() {
			iter.Last()
		}
		iter.Prev()
	}
}

type twoLevelBloomTombstoneState struct {
	c           *cache.Cache
	ch          *cache.Handle
	keys        [][]byte
	readers     [8][][]*sstable.Reader
	levelSlices [8][]manifest.LevelSlice
	indexFunc   func(twoLevelIndex bool, bloom bool, withTombstone bool) int
}

func (s *twoLevelBloomTombstoneState) close() {
	s.ch.Close()
	s.c.Unref()
	for _, r := range s.readers {
		for i := range r {
			for j := range r[i] {
				r[i][j].Close()
			}
		}
	}
}

func setupForTwoLevelBloomTombstone(b *testing.B, keyOffset int) twoLevelBloomTombstoneState {
	const blockSize = 32 << 10
	const restartInterval = 16
	const levelCount = 5

	c := NewCache(128 << 20 /* 128MB */)
	ch := c.NewHandle()
	var readers [8][][]*sstable.Reader
	var levelSlices [8][]manifest.LevelSlice
	var keys [][]byte
	indexFunc := func(twoLevelIndex bool, bloom bool, withTombstone bool) int {
		index := 0
		if twoLevelIndex {
			index = 4
		}
		if bloom {
			index += 2
		}
		if withTombstone {
			index++
		}
		return index
	}
	for _, twoLevelIndex := range []bool{false, true} {
		for _, bloom := range []bool{false, true} {
			for _, withTombstone := range []bool{false, true} {
				index := indexFunc(twoLevelIndex, bloom, withTombstone)
				levels := levelCount
				if withTombstone {
					levels = 1
				}
				readers[index], levelSlices[index], keys = buildLevelsForMergingIterSeqSeek(
					b, ch, blockSize, restartInterval, levels, nil, keyOffset, withTombstone, bloom, twoLevelIndex)
			}
		}
	}
	return twoLevelBloomTombstoneState{
		c:           c,
		ch:          ch,
		keys:        keys,
		readers:     readers,
		levelSlices: levelSlices,
		indexFunc:   indexFunc,
	}
}

// BenchmarkIteratorSeqSeekPrefixGENotFound exercises the case of SeekPrefixGE
// specifying monotonic keys all of which precede actual keys present in L6 of
// the DB. Moreover, with-tombstone=true exercises the sub-case where those
// actual keys are deleted using a range tombstone that has not physically
// deleted those keys due to the presence of a snapshot that needs to see
// those keys. This sub-case needs to be efficient in (a) avoiding iteration
// over all those deleted keys, including repeated iteration, (b) using the
// next optimization, since the seeks are monotonic.
func BenchmarkIteratorSeqSeekPrefixGENotFound(b *testing.B) {
	const keyOffset = 100000
	state := setupForTwoLevelBloomTombstone(b, keyOffset)
	defer state.close()
	readers := state.readers
	levelSlices := state.levelSlices
	indexFunc := state.indexFunc

	// We will not be seeking to the keys that were written but instead to
	// keys before the written keys. This is to validate that the optimization
	// to use Next still functions when mergingIter checks for the prefix
	// match, and that mergingIter can avoid iterating over all the keys
	// deleted by a range tombstone when there is no possibility of matching
	// the prefix.
	var keys [][]byte
	for i := 0; i < keyOffset; i++ {
		keys = append(keys, []byte(fmt.Sprintf("%08d", i)))
	}
	for _, skip := range []int{1, 2, 4} {
		for _, twoLevelIndex := range []bool{false, true} {
			for _, bloom := range []bool{false, true} {
				for _, withTombstone := range []bool{false, true} {
					b.Run(fmt.Sprintf("skip=%d/two-level=%t/bloom=%t/with-tombstone=%t",
						skip, twoLevelIndex, bloom, withTombstone),
						func(b *testing.B) {
							index := indexFunc(twoLevelIndex, bloom, withTombstone)
							readers := readers[index]
							levelSlices := levelSlices[index]
							m := buildMergingIter(readers, levelSlices)
							iter := Iterator{
								comparer: *testkeys.Comparer,
								merge:    DefaultMerger.Merge,
								iter:     m,
							}
							pos := 0
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								// When withTombstone=true, and prior to the
								// optimization to stop early due to a range
								// tombstone, the iteration would continue into the
								// next file, and not be able to use Next at the lower
								// level in the next SeekPrefixGE call. So we would
								// incur the cost of iterating over all the deleted
								// keys for every seek. Note that it is not possible
								// to do a noop optimization in Iterator for the
								// prefix case, unlike SeekGE/SeekLT, since we don't
								// know if the iterators inside mergingIter are all
								// appropriately positioned -- some may not be due to
								// bloom filters not matching.
								valid := iter.SeekPrefixGE(keys[pos])
								if valid {
									b.Fatalf("key should not be found")
								}
								pos += skip
								if pos >= keyOffset {
									pos = 0
								}
							}
							b.StopTimer()
							iter.Close()
						})
				}
			}
		}
	}
}

// BenchmarkIteratorSeqSeekPrefixGEFound exercises the case of SeekPrefixGE
// specifying monotonic keys that are present in L6 of the DB. Moreover,
// with-tombstone=true exercises the sub-case where those actual keys are
// deleted using a range tombstone that has not physically deleted those keys
// due to the presence of a snapshot that needs to see those keys. This
// sub-case needs to be efficient in (a) avoiding iteration over all those
// deleted keys, including repeated iteration, (b) using the next
// optimization, since the seeks are monotonic.
func BenchmarkIteratorSeqSeekPrefixGEFound(b *testing.B) {
	state := setupForTwoLevelBloomTombstone(b, 0)
	defer state.close()
	keys := state.keys
	readers := state.readers
	levelSlices := state.levelSlices
	indexFunc := state.indexFunc

	for _, skip := range []int{1, 2, 4} {
		for _, twoLevelIndex := range []bool{false, true} {
			for _, bloom := range []bool{false, true} {
				for _, withTombstone := range []bool{false, true} {
					b.Run(fmt.Sprintf("skip=%d/two-level=%t/bloom=%t/with-tombstone=%t",
						skip, twoLevelIndex, bloom, withTombstone),
						func(b *testing.B) {
							index := indexFunc(twoLevelIndex, bloom, withTombstone)
							readers := readers[index]
							levelSlices := levelSlices[index]
							m := buildMergingIter(readers, levelSlices)
							iter := Iterator{
								comparer: *testkeys.Comparer,
								merge:    DefaultMerger.Merge,
								iter:     m,
							}
							pos := 0
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								// When withTombstone=true, and prior to the
								// optimization to stop early due to a range
								// tombstone, the iteration would continue into the
								// next file, and not be able to use Next at the lower
								// level in the next SeekPrefixGE call. So we would
								// incur the cost of iterating over all the deleted
								// keys for every seek. Note that it is not possible
								// to do a noop optimization in Iterator for the
								// prefix case, unlike SeekGE/SeekLT, since we don't
								// know if the iterators inside mergingIter are all
								// appropriately positioned -- some may not be due to
								// bloom filters not matching.
								_ = iter.SeekPrefixGE(keys[pos])
								pos += skip
								if pos >= len(keys) {
									pos = 0
								}
							}
							b.StopTimer()
							iter.Close()
						})
				}
			}
		}
	}
}

// BenchmarkIteratorSeqSeekGEWithBounds is analogous to
// BenchmarkMergingIterSeqSeekGEWithBounds, except for using an Iterator,
// which causes it to exercise the end-to-end code path.
func BenchmarkIteratorSeqSeekGEWithBounds(b *testing.B) {
	const blockSize = 32 << 10
	const restartInterval = 16
	const levelCount = 5
	c := NewCache(128 << 20 /* 128MB */)
	defer c.Unref()
	for _, twoLevelIndex := range []bool{false, true} {
		b.Run(fmt.Sprintf("two-level=%t", twoLevelIndex),
			func(b *testing.B) {
				ch := c.NewHandle()
				defer ch.Close()
				readers, levelSlices, keys := buildLevelsForMergingIterSeqSeek(
					b, ch, blockSize, restartInterval, levelCount, nil, 0, /* keyOffset */
					false, false, twoLevelIndex)
				m := buildMergingIter(readers, levelSlices)
				iter := Iterator{
					comparer: *testkeys.Comparer,
					merge:    DefaultMerger.Merge,
					iter:     m,
				}
				keyCount := len(keys)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					pos := i % (keyCount - 1)
					iter.SetBounds(keys[pos], keys[pos+1])
					// SeekGE will return keys[pos].
					valid := iter.SeekGE(keys[pos])
					for valid {
						valid = iter.Next()
					}
					if iter.Error() != nil {
						b.Fatal(iter.Error().Error())
					}
				}
				iter.Close()
				for i := range readers {
					for j := range readers[i] {
						readers[i][j].Close()
					}
				}
			})
	}
}

func BenchmarkIteratorSeekGENoop(b *testing.B) {
	const blockSize = 32 << 10
	const restartInterval = 16
	const levelCount = 5
	const keyOffset = 10000

	c := NewCache(128 << 20 /* 128MB */)
	defer c.Unref()
	ch := c.NewHandle()
	defer ch.Close()
	readers, levelSlices, _ := buildLevelsForMergingIterSeqSeek(
		b, ch, blockSize, restartInterval, levelCount, nil, keyOffset, false, false, false)
	var keys [][]byte
	for i := 0; i < keyOffset; i++ {
		keys = append(keys, []byte(fmt.Sprintf("%08d", i)))
	}
	for _, withLimit := range []bool{false, true} {
		b.Run(fmt.Sprintf("withLimit=%t", withLimit), func(b *testing.B) {
			m := buildMergingIter(readers, levelSlices)
			iter := Iterator{
				comparer: *testkeys.Comparer,
				merge:    DefaultMerger.Merge,
				iter:     m,
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				pos := i % (len(keys) - 1)
				if withLimit {
					if iter.SeekGEWithLimit(keys[pos], keys[pos+1]) != IterAtLimit {
						b.Fatal("should be at limit")
					}
				} else {
					if !iter.SeekGE(keys[pos]) {
						b.Fatal("should be valid")
					}
				}
			}
			iter.Close()
		})
	}
	for i := range readers {
		for j := range readers[i] {
			readers[i][j].Close()
		}
	}
}

func BenchmarkBlockPropertyFilter(b *testing.B) {
	rng := rand.New(rand.NewPCG(0, uint64(time.Now().UnixNano())))
	for _, matchInterval := range []int{1, 10, 100, 1000} {
		b.Run(fmt.Sprintf("match-interval=%d", matchInterval), func(b *testing.B) {
			mem := vfs.NewMem()
			opts := &Options{
				FS:                 mem,
				FormatMajorVersion: FormatNewest,
				BlockPropertyCollectors: []func() BlockPropertyCollector{
					func() BlockPropertyCollector {
						return sstable.NewBlockIntervalCollector(
							"0", &testBlockIntervalMapper{numLength: 3}, nil, /* range key collector */
						)
					},
				},
			}
			d, err := Open("", opts)
			require.NoError(b, err)
			defer func() {
				require.NoError(b, d.Close())
			}()
			batch := d.NewBatch()
			const numKeys = 20 * 1000
			const valueSize = 1000
			for i := 0; i < numKeys; i++ {
				key := fmt.Sprintf("%06d%03d", i, i%matchInterval)
				value := randValue(valueSize, rng)
				require.NoError(b, batch.Set([]byte(key), value, nil))
			}
			require.NoError(b, batch.Commit(nil))
			require.NoError(b, d.Flush())
			require.NoError(b, d.Compact(context.Background(), nil, []byte{0xFF}, false))

			for _, filter := range []bool{false, true} {
				b.Run(fmt.Sprintf("filter=%t", filter), func(b *testing.B) {
					var iterOpts IterOptions
					if filter {
						iterOpts.PointKeyFilters = []BlockPropertyFilter{
							sstable.NewBlockIntervalFilter("0", uint64(0), uint64(1), nil),
						}
					}
					iter, _ := d.NewIter(&iterOpts)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						valid := iter.First()
						for valid {
							valid = iter.Next()
						}
					}
					b.StopTimer()
					require.NoError(b, iter.Close())
				})
			}
		})
	}
}

func TestRangeKeyMaskingRandomized(t *testing.T) {
	seed := *seed
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
		t.Logf("seed: %d", seed)
	}
	rng := rand.New(rand.NewPCG(0, seed))

	// Generate keyspace with point keys, and range keys which will
	// mask the point keys.
	var timestamps []int64
	for i := 0; i <= 100; i++ {
		timestamps = append(timestamps, rng.Int64N(1000))
	}

	ks := testkeys.Alpha(5)
	numKeys := 1000 + rng.IntN(9000)
	keys := make([][]byte, numKeys)
	keyTimeStamps := make([]int64, numKeys) // ts associated with the keys.
	for i := 0; i < numKeys; i++ {
		keys[i] = make([]byte, 5+testkeys.MaxSuffixLen)
		keyTimeStamps[i] = timestamps[rng.IntN(len(timestamps))]
		n := testkeys.WriteKeyAt(keys[i], ks, rng.Int64N(ks.Count()), keyTimeStamps[i])
		keys[i] = keys[i][:n]
	}

	numRangeKeys := rng.IntN(20)
	type rkey struct {
		start  []byte
		end    []byte
		suffix []byte
	}
	rkeys := make([]rkey, numRangeKeys)
	pointKeyHidden := make([]bool, numKeys)
	for i := 0; i < numRangeKeys; i++ {
		rkeys[i].start = make([]byte, 5)
		rkeys[i].end = make([]byte, 5)

		testkeys.WriteKey(rkeys[i].start[:5], ks, rng.Int64N(ks.Count()))
		testkeys.WriteKey(rkeys[i].end[:5], ks, rng.Int64N(ks.Count()))

		for bytes.Equal(rkeys[i].start[:5], rkeys[i].end[:5]) {
			testkeys.WriteKey(rkeys[i].end[:5], ks, rng.Int64N(ks.Count()))
		}

		if bytes.Compare(rkeys[i].start[:5], rkeys[i].end[:5]) > 0 {
			rkeys[i].start, rkeys[i].end = rkeys[i].end, rkeys[i].start
		}

		rkeyTimestamp := timestamps[rng.IntN(len(timestamps))]
		rkeys[i].suffix = []byte("@" + strconv.FormatInt(rkeyTimestamp, 10))

		// Each time we create a range key, check if the range key masks any
		// point keys.
		for j, pkey := range keys {
			if pointKeyHidden[j] {
				continue
			}

			if keyTimeStamps[j] >= rkeyTimestamp {
				continue
			}

			if testkeys.Comparer.Compare(pkey, rkeys[i].start) >= 0 &&
				testkeys.Comparer.Compare(pkey, rkeys[i].end) < 0 {
				pointKeyHidden[j] = true
			}
		}
	}

	// Define a simple base testOpts, and a randomized testOpts. The results
	// of iteration will be compared.
	type testOpts struct {
		levelOpts       []LevelOptions
		targetFileSizes []int64
		filter          func() BlockPropertyFilterMask
	}

	baseOpts := testOpts{
		levelOpts:       make([]LevelOptions, 7),
		targetFileSizes: make([]int64, 7),
	}
	for i := 0; i < len(baseOpts.levelOpts); i++ {
		baseOpts.targetFileSizes[i] = 1
		baseOpts.levelOpts[i].BlockSize = 1
	}

	randomOpts := testOpts{
		levelOpts: []LevelOptions{
			{BlockSize: 1 + rng.IntN(32<<10)},
		},
		targetFileSizes: []int64{int64(1 + rng.IntN(2<<20))}, // Vary the L0 file size.
	}
	if rng.IntN(2) == 0 {
		randomOpts.filter = func() BlockPropertyFilterMask {
			return sstable.NewTestKeysMaskingFilter()
		}
	}

	maxProcs := runtime.GOMAXPROCS(0)

	opts1 := &Options{
		FS:                         vfs.NewCrashableMem(),
		Comparer:                   testkeys.Comparer,
		FormatMajorVersion:         FormatNewest,
		CompactionConcurrencyRange: func() (int, int) { return 1, maxProcs/2 + 1 },
		BlockPropertyCollectors: []func() BlockPropertyCollector{
			sstable.NewTestKeysBlockPropertyCollector,
		},
	}
	copy(opts1.TargetFileSizes[:], baseOpts.targetFileSizes)
	copy(opts1.Levels[:], baseOpts.levelOpts)
	d1, err := Open("", opts1)
	require.NoError(t, err)

	opts2 := &Options{
		FS:                         vfs.NewCrashableMem(),
		Comparer:                   testkeys.Comparer,
		FormatMajorVersion:         FormatNewest,
		CompactionConcurrencyRange: func() (int, int) { return 1, maxProcs/2 + 1 },
		BlockPropertyCollectors: []func() BlockPropertyCollector{
			sstable.NewTestKeysBlockPropertyCollector,
		},
	}
	copy(opts2.TargetFileSizes[:], randomOpts.targetFileSizes)
	copy(opts2.Levels[:], randomOpts.levelOpts)
	d2, err := Open("", opts2)
	require.NoError(t, err)

	defer func() {
		if err := d1.Close(); err != nil {
			t.Fatal(err)
		}
		if err := d2.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	// Run test
	var batch1 *Batch
	var batch2 *Batch
	const keysPerBatch = 50
	for i := 0; i < numKeys; i++ {
		if i%keysPerBatch == 0 {
			if batch1 != nil {
				require.NoError(t, batch1.Commit(nil))
				require.NoError(t, batch2.Commit(nil))
			}
			batch1 = d1.NewBatch()
			batch2 = d2.NewBatch()
		}
		require.NoError(t, batch1.Set(keys[i], []byte{1}, nil))
		require.NoError(t, batch2.Set(keys[i], []byte{1}, nil))
	}

	for _, rkey := range rkeys {
		require.NoError(t, d1.RangeKeySet(rkey.start, rkey.end, rkey.suffix, nil, nil))
		require.NoError(t, d2.RangeKeySet(rkey.start, rkey.end, rkey.suffix, nil, nil))
	}

	// Scan the keyspace
	iter1Opts := IterOptions{
		KeyTypes: IterKeyTypePointsAndRanges,
		RangeKeyMasking: RangeKeyMasking{
			Suffix: []byte("@1000"),
			Filter: baseOpts.filter,
		},
	}

	iter2Opts := IterOptions{
		KeyTypes: IterKeyTypePointsAndRanges,
		RangeKeyMasking: RangeKeyMasking{
			Suffix: []byte("@1000"),
			Filter: randomOpts.filter,
		},
	}

	iter1, _ := d1.NewIter(&iter1Opts)
	iter2, _ := d2.NewIter(&iter2Opts)
	defer func() {
		if err := iter1.Close(); err != nil {
			t.Fatal(err)
		}
		if err := iter2.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	for valid1, valid2 := iter1.First(), iter2.First(); valid1 || valid2; valid1, valid2 = iter1.Next(), iter2.Next() {
		if valid1 != valid2 {
			t.Fatalf("iteration didn't produce identical results")
		}

		// Confirm exposed range key state is identical.
		hasP1, hasR1 := iter1.HasPointAndRange()
		hasP2, hasR2 := iter2.HasPointAndRange()
		if hasP1 != hasP2 || hasR1 != hasR2 {
			t.Fatalf("iteration didn't produce identical results")
		}
		if hasP1 && !bytes.Equal(iter1.Key(), iter2.Key()) {
			t.Fatalf("iteration didn't produce identical point keys: %s, %s", iter1.Key(), iter2.Key())
		}
		if hasR1 {
			// Confirm that the range key is the same.
			b1, e1 := iter1.RangeBounds()
			b2, e2 := iter2.RangeBounds()
			if !bytes.Equal(b1, b2) || !bytes.Equal(e1, e2) {
				t.Fatalf(
					"iteration didn't produce identical range keys: [%s, %s], [%s, %s]",
					b1, e1, b2, e2,
				)
			}

		}

		// Confirm that the returned point key wasn't hidden.
		for j, pkey := range keys {
			if bytes.Equal(iter1.Key(), pkey) && pointKeyHidden[j] {
				t.Fatalf("hidden point key was exposed %s %d", pkey, keyTimeStamps[j])
			}
		}
	}
}

func TestIteratorSeekPrefixGERandomized(t *testing.T) {
	seed := uint64(time.Now().UnixNano())
	t.Logf("seed: %d", seed)
	rng := rand.New(rand.NewPCG(seed, seed))
	ks := testkeys.Alpha(2)
	d := newTestkeysDatabase(t, ks, rng)
	defer func() { require.NoError(t, d.Close()) }()

	// Scan through the keys and construct a map from unique prefix to the
	// largest key with that prefix.
	m := make(map[string]string) // prefix -> largest user key
	iter, err := d.NewIter(nil)
	require.NoError(t, err)
	defer iter.Close()
	for valid := iter.First(); valid; valid = iter.Next() {
		prefix := string(d.opts.Comparer.Split.Prefix(iter.Key()))
		if _, ok := m[prefix]; !ok {
			m[string(prefix)] = string(iter.Key())
		}
	}
	// Perform a few seeks for random prefixes and ensure that result matches
	// the map constructed.
	for i := 0; i < 100; i++ {
		k := testkeys.Key(ks, rng.Int64N(ks.Count()))
		t.Logf("SeekPrefixGE(%q)", k)
		exists := iter.SeekPrefixGE(k)
		expectedKey, expectedOk := m[string(k)]
		require.Equal(t, expectedOk, exists)
		if exists {
			require.Equal(t, expectedKey, string(iter.Key()))
		}
	}
}

// BenchmarkIterator_RangeKeyMasking benchmarks a scan through a keyspace with
// 10,000 random suffixed point keys, and three range keys covering most of the
// keyspace. It varies the suffix of the range keys in subbenchmarks to exercise
// varying amounts of masking. This benchmark does configure a block-property
// filter, allowing for skipping blocks wholly contained within a range key and
// consisting of points all with a suffix lower than the range key's.
func BenchmarkIterator_RangeKeyMasking(b *testing.B) {
	const (
		prefixLen    = 20
		valueSize    = 1024
		batches      = 200
		keysPerBatch = 50
	)
	var alloc bytealloc.A
	rng := rand.New(rand.NewPCG(0, 1658872515083979000))
	keyBuf := make([]byte, prefixLen+testkeys.MaxSuffixLen)
	valBuf := make([]byte, valueSize)

	mem := vfs.NewMem()
	maxProcs := runtime.GOMAXPROCS(0)
	opts := &Options{
		FS:                         mem,
		Comparer:                   testkeys.Comparer,
		FormatMajorVersion:         FormatNewest,
		CompactionConcurrencyRange: func() (int, int) { return 1, maxProcs/2 + 1 },
		BlockPropertyCollectors: []func() BlockPropertyCollector{
			sstable.NewTestKeysBlockPropertyCollector,
		},
	}
	d, err := Open("", opts)
	require.NoError(b, err)

	keys := make([][]byte, keysPerBatch*batches)
	for bi := 0; bi < batches; bi++ {
		batch := d.NewBatch()
		for k := 0; k < keysPerBatch; k++ {
			randStr(keyBuf[:prefixLen], rng)
			suffix := rng.Int64N(100)
			suffixLen := testkeys.WriteSuffix(keyBuf[prefixLen:], suffix)
			randStr(valBuf[:], rng)

			var key []byte
			alloc, key = alloc.Copy(keyBuf[:prefixLen+suffixLen])
			keys[bi*keysPerBatch+k] = key
			require.NoError(b, batch.Set(key, valBuf[:], nil))
		}
		require.NoError(b, batch.Commit(nil))
	}

	// Wait for compactions to complete before starting benchmarks. We don't
	// want to benchmark while compactions are running.
	d.mu.Lock()
	for d.mu.compact.compactingCount > 0 {
		d.mu.compact.cond.Wait()
	}
	d.mu.Unlock()
	b.Log(d.Metrics().String())
	require.NoError(b, d.Close())

	// TODO(jackson): Benchmark lazy-combined iteration versus not.
	// TODO(jackson): Benchmark seeks.
	for _, rkSuffix := range []string{"@10", "@50", "@75", "@100"} {
		b.Run(fmt.Sprintf("range-keys-suffixes=%s", rkSuffix), func(b *testing.B) {
			// Clone the filesystem so that each subbenchmark may mutate state.
			opts := opts.Clone()
			opts.FS = vfs.NewMem()
			ok, err := vfs.Clone(mem, opts.FS, "", "")
			require.NoError(b, err)
			require.True(b, ok)
			d, err := Open("", opts)
			require.NoError(b, err)
			require.NoError(b, d.RangeKeySet([]byte("b"), []byte("e"), []byte(rkSuffix), nil, nil))
			require.NoError(b, d.RangeKeySet([]byte("f"), []byte("p"), []byte(rkSuffix), nil, nil))
			require.NoError(b, d.RangeKeySet([]byte("q"), []byte("z"), []byte(rkSuffix), nil, nil))
			require.NoError(b, d.Flush())

			// Populate 3 range keys, covering most of the keyspace, at the
			// given suffix.

			iterOpts := IterOptions{
				KeyTypes: IterKeyTypePointsAndRanges,
				RangeKeyMasking: RangeKeyMasking{
					Suffix: []byte("@100"),
					Filter: func() BlockPropertyFilterMask {
						return sstable.NewTestKeysMaskingFilter()
					},
				},
			}
			b.Run("forward", func(b *testing.B) {
				b.Run("seekprefix", func(b *testing.B) {
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						iter, _ := d.NewIter(&iterOpts)
						count := 0
						for j := 0; j < len(keys); j++ {
							if !iter.SeekPrefixGE(keys[j]) {
								b.Errorf("unable to find %q\n", keys[j])
							}
							if hasPoint, _ := iter.HasPointAndRange(); hasPoint {
								count++
							}
						}
						if err := iter.Close(); err != nil {
							b.Fatal(err)
						}
					}
				})
				b.Run("next", func(b *testing.B) {
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						iter, _ := d.NewIter(&iterOpts)
						count := 0
						for valid := iter.First(); valid; valid = iter.Next() {
							if hasPoint, _ := iter.HasPointAndRange(); hasPoint {
								count++
							}
						}
						if err := iter.Close(); err != nil {
							b.Fatal(err)
						}
					}
				})
			})
			b.Run("backward", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					iter, _ := d.NewIter(&iterOpts)
					count := 0
					for valid := iter.Last(); valid; valid = iter.Prev() {
						if hasPoint, _ := iter.HasPointAndRange(); hasPoint {
							count++
						}
					}
					if err := iter.Close(); err != nil {
						b.Fatal(err)
					}
				}
			})

			// Reset the benchmark state at the end of each run to remove the
			// range keys we wrote.
			b.StopTimer()
			require.NoError(b, d.Close())
		})
	}
}

func BenchmarkIteratorScan(b *testing.B) {
	const maxPrefixLen = 8
	keyBuf := make([]byte, maxPrefixLen+testkeys.MaxSuffixLen)
	rng := rand.New(rand.NewPCG(0, uint64(time.Now().UnixNano())))

	for _, keyCount := range []int64{100, 1000, 10000} {
		for _, readAmp := range []int{1, 3, 7, 10} {
			func() {
				opts := &Options{
					FS:                 vfs.NewMem(),
					FormatMajorVersion: FormatNewest,
				}
				opts.DisableAutomaticCompactions = true
				d, err := Open("", opts)
				require.NoError(b, err)
				defer func() { require.NoError(b, d.Close()) }()

				// Take the very large keyspace consisting of alphabetic
				// characters of lengths up to `maxPrefixLen` and reduce it down
				// to `keyCount` keys by picking every 1 key every `keyCount` keys.
				keys := testkeys.Alpha(maxPrefixLen)
				keys = keys.EveryN(keys.Count() / keyCount)
				if keys.Count() < keyCount {
					b.Fatalf("expected %d keys, found %d", keyCount, keys.Count())
				}

				// Portion the keys into `readAmp` overlapping key sets.
				for _, ks := range testkeys.Divvy(keys, int64(readAmp)) {
					batch := d.NewBatch()
					for i := int64(0); i < ks.Count(); i++ {
						n := testkeys.WriteKeyAt(keyBuf[:], ks, i, rng.Int64N(100))
						batch.Set(keyBuf[:n], keyBuf[:n], nil)
					}
					require.NoError(b, batch.Commit(nil))
					require.NoError(b, d.Flush())
				}
				// Each level is a sublevel.
				m := d.Metrics()
				require.Equal(b, readAmp, m.ReadAmp())

				for _, keyTypes := range []IterKeyType{IterKeyTypePointsOnly, IterKeyTypePointsAndRanges} {
					iterOpts := IterOptions{KeyTypes: keyTypes}
					b.Run(fmt.Sprintf("keys=%d,r-amp=%d,key-types=%s", keyCount, readAmp, keyTypes), func(b *testing.B) {
						for i := 0; i < b.N; i++ {
							b.StartTimer()
							iter, _ := d.NewIter(&iterOpts)
							valid := iter.First()
							for valid {
								valid = iter.Next()
							}
							b.StopTimer()
							require.NoError(b, iter.Close())
						}
					})
				}
			}()
		}
	}
}

func BenchmarkIteratorScanNextPrefix(b *testing.B) {
	setupBench := func(
		b *testing.B, maxKeysPerLevel, versCount, readAmp int, enableValueBlocks bool) *DB {
		keyBuf := make([]byte, readAmp+testkeys.MaxSuffixLen)
		opts := &Options{
			FS:                 vfs.NewMem(),
			Comparer:           testkeys.Comparer,
			FormatMajorVersion: FormatNewest,
		}
		opts.DisableAutomaticCompactions = true
		opts.Experimental.EnableValueBlocks = func() bool { return enableValueBlocks }
		d, err := Open("", opts)
		require.NoError(b, err)

		// Create `readAmp` levels. Prefixes in the top of the LSM are length 1.
		// Prefixes in the bottom of the LSM are length `readAmp`. Eg,:
		//
		//    a  b c...
		//    aa ab ac...
		//    aaa aab aac...
		//
		for l := readAmp; l > 0; l-- {
			ks := testkeys.Alpha(l)
			if step := ks.Count() / int64(maxKeysPerLevel); step > 1 {
				ks = ks.EveryN(step)
			}
			if ks.Count() > int64(maxKeysPerLevel) {
				ks = ks.Slice(0, int64(maxKeysPerLevel))
			}

			batch := d.NewBatch()
			for i := int64(0); i < ks.Count(); i++ {
				for v := 0; v < versCount; v++ {
					n := testkeys.WriteKeyAt(keyBuf[:], ks, i, int64(versCount-v+1))
					batch.Set(keyBuf[:n], keyBuf[:n], nil)
				}
			}
			require.NoError(b, batch.Commit(nil))
			require.NoError(b, d.Flush())
		}

		// Each level is a sublevel.
		m := d.Metrics()
		require.Equal(b, readAmp, m.ReadAmp())
		return d
	}

	for _, keysPerLevel := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("keysPerLevel=%d", keysPerLevel), func(b *testing.B) {
			for _, versionCount := range []int{1, 2, 10, 100} {
				b.Run(fmt.Sprintf("versions=%d", versionCount), func(b *testing.B) {
					for _, readAmp := range []int{1, 3, 7, 10} {
						b.Run(fmt.Sprintf("ramp=%d", readAmp), func(b *testing.B) {
							for _, enableValueBlocks := range []bool{false, true} {
								b.Run(fmt.Sprintf("value-blocks=%t", enableValueBlocks), func(b *testing.B) {
									d := setupBench(b, keysPerLevel, versionCount, readAmp, enableValueBlocks)
									defer func() { require.NoError(b, d.Close()) }()
									for _, keyTypes := range []IterKeyType{
										IterKeyTypePointsOnly, IterKeyTypePointsAndRanges} {
										b.Run(fmt.Sprintf("key-types=%s", keyTypes), func(b *testing.B) {
											iterOpts := IterOptions{KeyTypes: keyTypes}
											iter, _ := d.NewIter(&iterOpts)
											var valid bool
											b.ResetTimer()
											for i := 0; i < b.N; i++ {
												if !valid {
													valid = iter.First()
													if !valid {
														b.Fatalf("iter must be valid")
													}
												} else {
													valid = iter.NextPrefix()
												}
											}
											b.StopTimer()
											require.NoError(b, iter.Close())
										})
									}
								})
							}
						})
					}
				})
			}
		})
	}
}

func BenchmarkCombinedIteratorSeek(b *testing.B) {
	for _, withRangeKey := range []bool{false, true} {
		b.Run(fmt.Sprintf("range-key=%t", withRangeKey), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(0, 1658872515083979000))
			ks := testkeys.Alpha(1)
			opts := &Options{
				FS:                 vfs.NewMem(),
				Comparer:           testkeys.Comparer,
				FormatMajorVersion: FormatNewest,
			}
			d, err := Open("", opts)
			require.NoError(b, err)
			defer func() { require.NoError(b, d.Close()) }()

			keys := make([][]byte, ks.Count())
			for i := int64(0); i < ks.Count(); i++ {
				keys[i] = testkeys.Key(ks, i)
				var val [40]byte
				for j := range val {
					val[j] = byte(rng.Uint32())
				}
				require.NoError(b, d.Set(keys[i], val[:], nil))
			}
			if withRangeKey {
				require.NoError(b, d.RangeKeySet([]byte("a"), []byte{'z', 0x00}, []byte("@5"), nil, nil))
			}

			batch := d.NewIndexedBatch()
			defer batch.Close()

			for _, useBatch := range []bool{false, true} {
				b.Run(fmt.Sprintf("batch=%t", useBatch), func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						iterOpts := IterOptions{KeyTypes: IterKeyTypePointsAndRanges}
						var it *Iterator
						if useBatch {
							it, _ = batch.NewIter(&iterOpts)
						} else {
							it, _ = d.NewIter(&iterOpts)
						}
						for j := 0; j < len(keys); j++ {
							if !it.SeekGE(keys[j]) {
								b.Errorf("key %q missing", keys[j])
							}
						}
						require.NoError(b, it.Close())
					}
				})
			}
		})
	}
}

// BenchmarkCombinedIteratorSeek_Bounded benchmarks a bounded iterator that
// performs repeated seeks over 5% of the middle of a keyspace covered by a
// range key that's fragmented across hundreds of files. The iterator bounds
// should prevent defragmenting beyond the iterator's bounds.
func BenchmarkCombinedIteratorSeek_Bounded(b *testing.B) {
	d, keys := buildFragmentedRangeKey(b, 1658872515083979000)

	var lower = len(keys) / 2
	var upper = len(keys)/2 + len(keys)/20 // 5%
	iterOpts := IterOptions{
		KeyTypes:   IterKeyTypePointsAndRanges,
		LowerBound: keys[lower],
		UpperBound: keys[upper],
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		it, _ := d.NewIter(&iterOpts)
		for j := lower; j < upper; j++ {
			if !it.SeekGE(keys[j]) {
				b.Errorf("key %q missing", keys[j])
			}
		}
		require.NoError(b, it.Close())
	}
}

// BenchmarkCombinedIteratorSeekPrefix benchmarks an iterator that
// performs repeated prefix seeks over 5% of the middle of a keyspace covered by a
// range key that's fragmented across hundreds of files. The seek prefix should
// avoid defragmenting beyond the seek prefixes.
func BenchmarkCombinedIteratorSeekPrefix(b *testing.B) {
	d, keys := buildFragmentedRangeKey(b, 1658872515083979000)

	var lower = len(keys) / 2
	var upper = len(keys)/2 + len(keys)/20 // 5%
	iterOpts := IterOptions{
		KeyTypes: IterKeyTypePointsAndRanges,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		it, _ := d.NewIter(&iterOpts)
		for j := lower; j < upper; j++ {
			if !it.SeekPrefixGE(keys[j]) {
				b.Errorf("key %q missing", keys[j])
			}
		}
		require.NoError(b, it.Close())
	}
}

func buildFragmentedRangeKey(b testing.TB, seed uint64) (d *DB, keys [][]byte) {
	rng := rand.New(rand.NewPCG(0, seed))
	ks := testkeys.Alpha(2)
	opts := &Options{
		FS:                        vfs.NewMem(),
		Comparer:                  testkeys.Comparer,
		FormatMajorVersion:        FormatNewest,
		L0CompactionFileThreshold: 1,
	}
	opts.EnsureDefaults()
	for l := range opts.TargetFileSizes {
		opts.TargetFileSizes[l] = 1
	}
	var err error
	d, err = Open("", opts)
	require.NoError(b, err)

	keys = make([][]byte, ks.Count())
	for i := int64(0); i < ks.Count(); i++ {
		keys[i] = testkeys.Key(ks, i)
	}
	for i := 0; i < len(keys); i++ {
		var val [40]byte
		for j := range val {
			val[j] = byte(rng.Uint32())
		}
		require.NoError(b, d.Set(keys[i], val[:], nil))
		if i < len(keys)-1 {
			require.NoError(b, d.RangeKeySet(keys[i], keys[i+1], []byte("@5"), nil, nil))
		}
		require.NoError(b, d.Flush())
	}

	d.mu.Lock()
	for d.mu.compact.compactingCount > 0 {
		d.mu.compact.cond.Wait()
	}
	v := d.mu.versions.currentVersion()
	d.mu.Unlock()
	require.GreaterOrEqualf(b, v.Levels[numLevels-1].Len(),
		700, "expect many (≥700) L6 files but found %d", v.Levels[numLevels-1].Len())
	return d, keys
}

// BenchmarkSeekPrefixTombstones benchmarks a SeekPrefixGE into the beginning of
// a series of sstables containing exclusively range tombstones. Previously,
// such a seek would next through all the tombstone files until it arrived at a
// point key or exhausted the level's files. The SeekPrefixGE should not next
// beyond the files that contain the prefix.
//
// See cockroachdb/cockroach#89327.
func BenchmarkSeekPrefixTombstones(b *testing.B) {
	o := &Options{
		FS:                 vfs.NewMem(),
		Comparer:           testkeys.Comparer,
		FormatMajorVersion: FormatNewest,
	}
	o.EnsureDefaults()
	d, err := Open("", o)
	require.NoError(b, err)
	defer func() { require.NoError(b, d.Close()) }()
	wOpts := o.MakeWriterOptions(numLevels-1, d.TableFormat())

	// Keep a snapshot open for the duration of the test to prevent elision-only
	// compactions from removing the ingested files containing exclusively
	// elidable tombstones.
	defer d.NewSnapshot().Close()

	ks := testkeys.Alpha(2)
	for i := int64(0); i < ks.Count()-1; i++ {
		func() {
			filename := fmt.Sprintf("ext%2d", i)
			f, err := o.FS.Create(filename, vfs.WriteCategoryUnspecified)
			require.NoError(b, err)
			w := sstable.NewWriter(objstorageprovider.NewFileWritable(f), wOpts)
			require.NoError(b, w.DeleteRange(testkeys.Key(ks, i), testkeys.Key(ks, i+1)))
			require.NoError(b, w.Close())
			require.NoError(b, d.Ingest(context.Background(), []string{filename}))
		}()
	}

	d.mu.Lock()
	require.Equal(b, int64(ks.Count()-1), d.mu.versions.metrics.Levels[numLevels-1].TablesCount)
	d.mu.Unlock()

	seekKey := testkeys.Key(ks, 1)
	iter, _ := d.NewIter(nil)
	defer iter.Close()
	b.ResetTimer()
	defer b.StopTimer()
	for i := 0; i < b.N; i++ {
		iter.SeekPrefixGE(seekKey)
	}
}

func waitForCompactionsAndTableStats(d *DB) {
	d.mu.Lock()
	// NB: Wait for table stats because some compaction types rely
	// on table stats to be collected.
	d.waitTableStats()
	for d.mu.compact.compactingCount > 0 {
		d.mu.compact.cond.Wait()
		d.waitTableStats()
	}
	d.mu.Unlock()
}

// BenchmarkPointDeletedSwath benchmarks iterator operations on large-ish
// (hundreds of MBs) databases containing broad swaths of keys removed by point
// tombstones.
func BenchmarkPointDeletedSwath(b *testing.B) {
	const maxKeyLen = 5
	ks := testkeys.Alpha(maxKeyLen)

	opts := func() *Options {
		o := &Options{
			FS:                 vfs.NewMem(),
			Comparer:           testkeys.Comparer,
			FormatMajorVersion: FormatNewest,
		}
		o.EnsureDefaults()
		return o
	}
	type iteratorOp struct {
		name string
		fn   func(*Iterator, testkeys.Keyspace, *rand.Rand)
	}
	var iterKeyBuf [maxKeyLen]byte

	iterOps := []iteratorOp{
		{
			name: "seek-prefix-ge", fn: func(iter *Iterator, ks testkeys.Keyspace, rng *rand.Rand) {
				n := testkeys.WriteKey(iterKeyBuf[:], ks, int64(rng.IntN(int(ks.Count()))))
				_ = iter.SeekPrefixGE(iterKeyBuf[:n])
			},
		},
		{
			name: "seek-ge", fn: func(iter *Iterator, ks testkeys.Keyspace, rng *rand.Rand) {
				n := testkeys.WriteKey(iterKeyBuf[:], ks, int64(rng.IntN(int(ks.Count()))))
				_ = iter.SeekGE(iterKeyBuf[:n])
			},
		},
		{
			name: "iterate", fn: func(iter *Iterator, ks testkeys.Keyspace, rng *rand.Rand) {
				valid := iter.Next()
				if !valid {
					iter.First()
				}
			},
		},
	}

	// Populate an initial database with point keys at every key in the `ks`
	// keyspace.
	populated := withStateSetup(b, vfs.NewMem(), opts(), populateKeyspaceSetup(ks))
	for _, gapLength := range []int{100, 1_000, 10_000, 100_000, 200_000, 400_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000} {
		b.Run(fmt.Sprintf("gap=%d", gapLength), func(b *testing.B) {
			// Extend the `populated` initial database with DELs deleting all
			// the middle keys in the keyspace in a contiguous swath of
			// `gapLength` keys.
			gapDeleted := withStateSetup(b, populated, opts(), deleteGapSetup(ks, gapLength))

			for _, op := range iterOps {
				b.Run(op.name, func(b *testing.B) {
					// Run each instance of the test in a fresh DB constructed
					// from `compacted`. This ensures background compactions
					// from one iterator operation don't affect another iterator
					// option.
					withStateSetup(b, gapDeleted, opts(), func(_ testing.TB, d *DB) {
						rng := rand.New(rand.NewPCG(0, 1) /* fixed seed */)
						iter, err := d.NewIter(nil)
						require.NoError(b, err)
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							op.fn(iter, ks, rng)
						}
						b.StopTimer()
						require.NoError(b, iter.Close())
					})
				})
			}
		})
	}
}

func withStateSetup(
	t testing.TB, initial vfs.FS, opts *Options, setup func(testing.TB, *DB),
) vfs.FS {
	ok, err := vfs.Clone(initial, opts.FS, "", "", vfs.CloneSync)
	require.NoError(t, err)
	require.True(t, ok)
	d, err := Open("", opts)
	require.NoError(t, err)
	defer func() { require.NoError(t, d.Close()) }()
	setup(t, d)
	return opts.FS
}

func populateKeyspaceSetup(ks testkeys.Keyspace) func(testing.TB, *DB) {
	const valSize = 256
	return func(t testing.TB, d *DB) {
		t.Logf("Populating keyspace with %d keys, each with %d-byte values", ks.Count(), valSize)
		// Parallelize population by divvying up the keyspace.
		var grp errgroup.Group
		loadKeyspaces := testkeys.Divvy(ks, 20)
		var progress atomic.Uint64
		for l := 0; l < len(loadKeyspaces); l++ {
			l := l
			grp.Go(func() error {
				rng := rand.New(rand.NewPCG(1, 1))
				batch := d.NewBatch()
				key := make([]byte, ks.MaxLen())
				var val [valSize]byte
				for i := int64(0); i < loadKeyspaces[l].Count(); i++ {
					for j := range val {
						val[j] = byte(rng.Uint32())
					}
					n := testkeys.WriteKey(key, loadKeyspaces[l], i)
					if err := batch.Set(key[:n], val[:], nil); err != nil {
						return err
					}
					if batch.Len() >= 10<<10 /* 10 kib */ {
						count := batch.Count()
						require.NoError(t, batch.Commit(NoSync))
						if newTotal := progress.Add(uint64(count)); (newTotal / (uint64(ks.Count()) / 100)) != (newTotal-uint64(count))/uint64(ks.Count()/100) {
							t.Logf("%.1f%% populated", 100.0*(float64(newTotal)/float64(ks.Count())))
						}
						batch = d.NewBatch()
						d.AsyncFlush()
					}
				}
				if !batch.Empty() {
					return batch.Commit(NoSync)
				}
				return nil
			})
		}
		require.NoError(t, grp.Wait())
	}
}

func deleteGapSetup(ks testkeys.Keyspace, gapLength int) func(testing.TB, *DB) {
	return func(t testing.TB, d *DB) {
		midpoint := ks.Count() / 2
		gapStart := midpoint - int64(gapLength/2)
		gapEnd := midpoint + int64(gapLength/2+(gapLength%2))

		batch := d.NewBatch()
		key := make([]byte, ks.MaxLen())
		for i := gapStart; i <= gapEnd; i++ {
			n := testkeys.WriteKey(key, ks, i)
			if err := batch.Delete(key[:n], nil); err != nil {
				t.Fatal(err)
			}
			if batch.Len() >= 10<<10 /* 10 kib */ {
				if err := batch.Commit(NoSync); err != nil {
					t.Fatal(err)
				}
				batch = d.NewBatch()
			}
		}
		if !batch.Empty() {
			if err := batch.Commit(NoSync); err != nil {
				t.Fatal(err)
			}
		}
		if err := d.Flush(); err != nil {
			t.Fatal(err)
		}
		waitForCompactionsAndTableStats(d)
	}
}

func runBenchmarkQueueWorkload(b *testing.B, deleteRatio float32, initOps int, valueSize int) {
	const queueCount = 8
	// These should be large enough to assign a unique key to each item in the
	// queue.
	const maxQueueIDLen = 1
	const maxItemLen = 7
	const maxKeyLen = maxQueueIDLen + 1 + maxItemLen
	queueIDKeyspace := testkeys.Alpha(maxQueueIDLen)
	itemKeyspace := testkeys.Alpha(maxItemLen)
	key := make([]byte, maxKeyLen)
	val := make([]byte, valueSize)
	rng := rand.New(rand.NewPCG(0, uint64(time.Now().UnixNano())))

	getKey := func(q int, i int) []byte {
		n := testkeys.WriteKey(key, queueIDKeyspace, int64(q))
		key[n] = '/'
		prefixLen := n + 1
		n = testkeys.WriteKey(key[prefixLen:], itemKeyspace, int64(i))
		return key[:prefixLen+n]
	}

	type Queue struct {
		start int
		end   int // exclusive
	}
	var queues = make([]*Queue, queueCount)
	for i := 0; i < queueCount; i++ {
		queues[i] = &Queue{}
	}

	o := &Options{
		Cache:              cache.New(0), // disable cache
		DisableWAL:         true,
		FS:                 vfs.NewMem(),
		Comparer:           testkeys.Comparer,
		FormatMajorVersion: FormatNewest,
	}
	o.EnsureDefaults()

	d, err := Open("", o)
	require.NoError(b, err)

	processQueueOnce := func(batch *Batch) {
		for {
			// Randomly pick a queue to process.
			q := rng.IntN(queueCount)
			queue := queues[q]

			isDelete := rng.Float32() < deleteRatio

			if isDelete {
				// Only process the queue if it's not empty. Otherwise, retry
				// with a different queue.
				if queue.start != queue.end {
					require.NoError(b, batch.Delete(getKey(q, queue.start), nil))
					queue.start = (queue.start + 1) % int(itemKeyspace.Count())
					break
				}
			} else {
				// Append to the queue.
				require.NoError(b, batch.Set(getKey(q, queue.end), val, nil))
				queue.end = (queue.end + 1) % int(itemKeyspace.Count())
				break
			}
		}
	}

	// First, process queues initialOps times.
	batch := d.NewBatch()
	for i := 0; i < initOps; i++ {
		processQueueOnce(batch)
		// Use a large batch size to speed up initialization.
		if batch.Len() >= 10<<24 /* 167 MiB */ {
			require.NoError(b, batch.Commit(NoSync))
			batch = d.NewBatch()
		}
	}
	require.NoError(b, batch.Commit(NoSync))
	// Manually flush in case the last batch was small.
	_, err = d.AsyncFlush()
	require.NoError(b, err)

	waitForCompactionsAndTableStats(d)

	// Log the number of tombstones and live keys in each level after
	// background compactions are complete.
	b.Log("LSM after compactions:")
	firstIter, _ := d.NewIter(nil)
	firstIter.First()
	lastIter, _ := d.NewIter(nil)
	lastIter.Last()
	stats, _ := d.ScanStatistics(context.Background(), firstIter.Key(), lastIter.Key(), ScanStatisticsOptions{})
	require.NoError(b, firstIter.Close())
	require.NoError(b, lastIter.Close())
	metrics := d.Metrics()
	for i := 0; i < numLevels; i++ {
		numTombstones := stats.Levels[i].KindsCount[base.InternalKeyKindDelete]
		numSets := stats.Levels[i].KindsCount[base.InternalKeyKindSet]
		numTables := metrics.Levels[i].TablesCount
		if numSets > 0 {
			b.Logf("L%d: %d tombstones, %d sets, %d sstables\n", i, numTombstones, numSets, numTables)
		}
	}

	// Seek to the start of each queue.
	b.Run("seek", func(b *testing.B) {
		iter, _ := d.NewIter(nil)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for q := 0; q < queueCount; q++ {
				iter.SeekGE(getKey(q, 0))
			}
		}
		b.StopTimer()
		require.NoError(b, iter.Close())
	})

	require.NoError(b, d.Close())
}

// BenchmarkQueueWorkload benchmarks a workload consisting of multiple queues
// that are all being processed at the same time. Processing a queue entails
// either appending to the end of the queue (a Set operation) or deleting from
// the start of the queue (a Delete operation). The goal is to detect cases
// where we see a large buildup of point tombstones at the beginning of each
// queue, which leads to the slowdown of SeekGE(<start of queue>). To that end,
// the test subbenchmarks a series of configurations that each 1) process the
// queues a certain number of times and then 2) benchmark both the queue
// processing throughput and SeekGE performance. See
// https://github.com/facebook/rocksdb/wiki/Implement-Queue-Service-Using-RocksDB
// for more information.
func BenchmarkQueueWorkload(b *testing.B) {
	// The portion of processing ops that will be deletes for each subbenchmark.
	var deleteRatios = []float32{0.1, 0.3, 0.5}
	// The number of times queues will be processed before running each
	// subbenchmark.
	var initOps = []int{400_000, 800_000, 1_200_000, 2_000_000, 3_500_000, 5_000_000, 7_500_000}
	// We vary the value size to identify how compaction behaves when the
	// relative sizes of tombstones and the keys they delete are different.
	var valueSizes = []int{128, 2048}

	for _, deleteRatio := range deleteRatios {
		for _, valueSize := range valueSizes {
			for _, numInitOps := range initOps {
				b.Run(fmt.Sprintf("initial_ops=%d/deleteRatio=%.2f/valueSize=%d", numInitOps, deleteRatio, valueSize), func(b *testing.B) {
					runBenchmarkQueueWorkload(b, deleteRatio, numInitOps, valueSize)
				})
			}
		}
	}
}
