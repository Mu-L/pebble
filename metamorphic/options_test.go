// Copyright 2022 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package metamorphic

import (
	"context"
	"fmt"
	"io/fs"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble"
	"github.com/cockroachdb/pebble/internal/testkeys"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/kr/pretty"
	"github.com/stretchr/testify/require"
)

func TestSetupInitialState(t *testing.T) {
	// Construct a small database in the test's TempDir.
	initialStatePath := t.TempDir()
	initialDataPath := vfs.Default.PathJoin(initialStatePath, "data")
	{
		d, err := pebble.Open(initialDataPath, &pebble.Options{})
		require.NoError(t, err)
		const maxKeyLen = 2
		ks := testkeys.Alpha(maxKeyLen)
		var key [maxKeyLen]byte
		for i := int64(0); i < ks.Count(); i++ {
			n := testkeys.WriteKey(key[:], ks, i)
			require.NoError(t, d.Set(key[:n], key[:n], pebble.NoSync))
			if i%100 == 0 {
				require.NoError(t, d.Flush())
			}
		}
		require.NoError(t, d.Close())
	}
	ls, err := vfs.Default.List(initialStatePath)
	require.NoError(t, err)

	// setupInitialState with an initial state path set to the test's TempDir
	// should populate opts.opts.FS with the directory's contents.
	opts := &TestOptions{
		Opts:             defaultOptions(TestkeysKeyFormat),
		initialStatePath: initialStatePath,
		initialStateDesc: "test",
	}
	require.NoError(t, setupInitialState("data", opts))
	copied, err := opts.Opts.FS.List("")
	require.NoError(t, err)
	require.ElementsMatch(t, ls, copied)
}

func TestOptionsRoundtrip(t *testing.T) {
	// Some fields must be ignored to avoid spurious diffs.
	ignorePrefixes := []string{
		// Pointers
		"Cache:",
		"Cache.",
		"FS:",
		"KeySchemas[",
		"FileCache:",
		"Experimental.CompactionScheduler",
		// Function pointers
		"BlockPropertyCollectors:",
		"EventListener:",
		"CompactionConcurrencyRange:",
		"MaxConcurrentDownloads:",
		"Experimental.CompactionGarbageFractionForMaxConcurrency:",
		"Experimental.DisableIngestAsFlushable:",
		"Experimental.EnableColumnarBlocks:",
		"Experimental.EnableValueBlocks:",
		"Experimental.IneffectualSingleDeleteCallback:",
		"Experimental.IngestSplit:",
		"Experimental.RemoteStorage:",
		"Experimental.SingleDeleteInvariantViolationCallback:",
		"Experimental.EnableDeleteOnlyCompactionExcises:",
		"Experimental.ValueSeparationPolicy:",
		"Levels[0].Compression:",
		"Levels[1].Compression:",
		"Levels[2].Compression:",
		"Levels[3].Compression:",
		"Levels[4].Compression:",
		"Levels[5].Compression:",
		"Levels[6].Compression:",
		"WALFailover.FailoverOptions.UnhealthyOperationLatencyThreshold:",
		// Floating points
		"Experimental.PointTombstoneWeight:",
		"Experimental.MultiLevelCompactionHeuristic.AddPropensity",
		"Experimental.MultiLevelCompactionHeuristic",
	}

	checkOptions := func(t *testing.T, o *TestOptions) {
		s := optionsToString(o)
		t.Logf("Serialized options:\n%s\n", s)

		parsed := defaultTestOptions(TestkeysKeyFormat)
		require.NoError(t, parseOptions(parsed, s, nil))
		got := optionsToString(parsed)
		require.Equal(t, s, got)
		t.Logf("Re-serialized options:\n%s\n", got)

		// In some options, the closure obscures the underlying value. Check
		// that the return values are equal.
		expectEqualFn(t, o.Opts.Experimental.EnableValueBlocks, parsed.Opts.Experimental.EnableValueBlocks)
		expectEqualFn(t, o.Opts.Experimental.DisableIngestAsFlushable, parsed.Opts.Experimental.DisableIngestAsFlushable)
		expectEqualFn(t, o.Opts.Experimental.IngestSplit, parsed.Opts.Experimental.IngestSplit)
		expectEqualFn(t, o.Opts.Experimental.CompactionGarbageFractionForMaxConcurrency, parsed.Opts.Experimental.CompactionGarbageFractionForMaxConcurrency)
		expectEqualFn(t, o.Opts.Experimental.ValueSeparationPolicy, parsed.Opts.Experimental.ValueSeparationPolicy)

		expBaseline, expUpper := o.Opts.CompactionConcurrencyRange()
		parsedBaseline, parsedUpper := parsed.Opts.CompactionConcurrencyRange()
		require.Equal(t, expBaseline, parsedBaseline)
		require.Equal(t, expUpper, parsedUpper)

		require.Equal(t, o.Opts.MaxConcurrentDownloads(), parsed.Opts.MaxConcurrentDownloads())
		require.Equal(t, len(o.Opts.BlockPropertyCollectors), len(parsed.Opts.BlockPropertyCollectors))

		diff := pretty.Diff(o.Opts, parsed.Opts)
		cleaned := diff[:0]
		for _, d := range diff {
			var ignored bool
			for _, prefix := range ignorePrefixes {
				if strings.HasPrefix(d, prefix) {
					ignored = true
					break
				}
			}
			if !ignored {
				cleaned = append(cleaned, d)
			}
		}
		require.Equal(t, diff[:0], cleaned, "before:\n%#v\nafter:\n%#v\n", o.Opts, parsed.Opts)
	}

	standard := standardOptions(TestkeysKeyFormat)
	for i := range standard {
		t.Run(fmt.Sprintf("standard-%03d", i), func(t *testing.T) {
			checkOptions(t, standard[i])
		})
	}
	rng := rand.New(rand.NewPCG(0, uint64(time.Now().UnixNano())))
	for i := 0; i < 100; i++ {
		t.Run(fmt.Sprintf("random-%03d", i), func(t *testing.T) {
			o := RandomOptions(rng, TestkeysKeyFormat, RandomOptionsCfg{})
			checkOptions(t, o)
		})
	}
}

// expectEqualFn verifies that both expected and actual are nil, or they are
// both non-nil and return equal values.
func expectEqualFn[T any](t *testing.T, expected func() T, actual func() T) {
	t.Helper()
	if expected == nil {
		require.Nil(t, actual)
	} else {
		require.NotNil(t, actual)
		ev, av := expected(), actual()
		expectEqualValue(t, reflect.ValueOf(ev).Interface(), reflect.ValueOf(av).Interface())
	}
}

func expectEqualValue(t *testing.T, expected any, actual any) {
	t.Helper()
	typ := reflect.TypeOf(expected)
	switch typ.Kind() {
	case reflect.Float64:
		require.InDelta(t, expected, actual, 1e-2)
	case reflect.Struct:
		for i := range typ.NumField() {
			expectEqualValue(t, reflect.ValueOf(expected).Field(i).Interface(),
				reflect.ValueOf(actual).Field(i).Interface())
		}
	default:
		require.Equal(t, expected, actual)
	}
}

// TestBlockPropertiesParse ensures that the testkeys block property collector
// is in use by default. It runs a single OPTIONS run of the metamorphic tests
// and scans the resulting data directory to ensure there's at least one sstable
// with the property. It runs the test with the archive cleaner to avoid any
// flakiness from small working sets of keys.
func TestBlockPropertiesParse(t *testing.T) {
	const fixedSeed = 1
	const numOps = 10_000
	metaDir := t.TempDir()

	rng := rand.New(rand.NewPCG(0, fixedSeed))
	km := newKeyManager(1 /* numInstances */, TestkeysKeyFormat)
	g := newGenerator(rng, presetConfigs[0], km)
	ops := g.generate(numOps)
	opsPath := filepath.Join(metaDir, "ops")
	formattedOps := formatOps(km.kf, ops)
	require.NoError(t, os.WriteFile(opsPath, []byte(formattedOps), 0644))

	runDir := filepath.Join(metaDir, "run")
	require.NoError(t, os.MkdirAll(runDir, os.ModePerm))
	optionsPath := filepath.Join(runDir, "OPTIONS")
	opts := defaultTestOptions(TestkeysKeyFormat)
	opts.Opts.EnsureDefaults()
	opts.Opts.Cleaner = pebble.ArchiveCleaner{}
	optionsStr := optionsToString(opts)
	require.NoError(t, os.WriteFile(optionsPath, []byte(optionsStr), 0644))

	RunOnce(t, runDir, fixedSeed, filepath.Join(runDir, "history"), TestkeysKeyFormat, KeepData{})
	var foundTableBlockProperty bool
	require.NoError(t, filepath.Walk(filepath.Join(runDir, "data"),
		func(path string, info fs.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if filepath.Ext(path) != ".sst" {
				return nil
			}
			f, err := vfs.Default.Open(path)
			if err != nil {
				return err
			}
			readable, err := sstable.NewSimpleReadable(f)
			if err != nil {
				return err
			}
			r, err := sstable.NewReader(context.Background(), readable, opts.Opts.MakeReaderOptions())
			if err != nil {
				return errors.CombineErrors(err, readable.Close())
			}
			_, ok := r.UserProperties[opts.Opts.BlockPropertyCollectors[0]().Name()]
			foundTableBlockProperty = foundTableBlockProperty || ok
			return r.Close()
		}))
	require.True(t, foundTableBlockProperty)
}

func TestCustomOptionParser(t *testing.T) {
	customOptionParsers := map[string]func(string) (CustomOption, bool){
		"foo": func(value string) (CustomOption, bool) {
			return testCustomOption{name: "foo", value: value}, true
		},
	}

	o1 := defaultTestOptions(TestkeysKeyFormat)
	o2 := defaultTestOptions(TestkeysKeyFormat)

	require.NoError(t, parseOptions(o1, `
[TestOptions]
  foo=bar
`, customOptionParsers))
	require.NoError(t, parseOptions(o2, optionsToString(o1), customOptionParsers))

	for _, o := range []*TestOptions{o1, o2} {
		require.Equal(t, 1, len(o.CustomOpts))
		require.Equal(t, "foo", o.CustomOpts[0].Name())
		require.Equal(t, "bar", o.CustomOpts[0].Value())
	}
}

type testCustomOption struct {
	name, value string
}

func (o testCustomOption) Name() string                { return o.name }
func (o testCustomOption) Value() string               { return o.value }
func (o testCustomOption) Close(*pebble.Options) error { return nil }
func (o testCustomOption) Open(*pebble.Options) error  { return nil }
