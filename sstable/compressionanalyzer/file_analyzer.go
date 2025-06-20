// Copyright 2025 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package compressionanalyzer

import (
	"context"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/objstorage"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/sstable/block"
	"github.com/cockroachdb/pebble/sstable/block/blockkind"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/cockroachdb/tokenbucket"
)

// FileAnalyzer is used to analyze blocks in sstable files.
//
// TODO(radu): add support for blob files.
type FileAnalyzer struct {
	*BlockAnalyzer

	readLimiter *tokenbucket.TokenBucket
	sstReadOpts sstable.ReaderOptions
}

func NewFileAnalyzer(
	readLimiter *tokenbucket.TokenBucket, sstReadOpts sstable.ReaderOptions,
) *FileAnalyzer {
	if sstReadOpts.CacheOpts.CacheHandle != nil {
		// We do not support using a cache here (we don't properly populate the
		// block metadata).
		panic("sstReadOpts.CacheOpts.CacheHandle must be nil")
	}
	return &FileAnalyzer{
		BlockAnalyzer: NewBlockAnalyzer(),
		readLimiter:   readLimiter,
		sstReadOpts:   sstReadOpts,
	}
}

func (fa *FileAnalyzer) Close() {
	if fa.BlockAnalyzer != nil {
		fa.BlockAnalyzer.Close()
	}
	*fa = FileAnalyzer{}
}

// SSTable analyzes the blocks in an sstable file.
func (fa *FileAnalyzer) SSTable(ctx context.Context, fs vfs.FS, path string) error {
	// Obtain the layout.
	file, err := fs.Open(path)
	if err != nil {
		return err
	}
	readable, err := sstable.NewSimpleReadable(file)
	if err != nil {
		_ = file.Close()
		return err
	}
	r, err := sstable.NewReader(ctx, readable, fa.sstReadOpts)
	if err != nil {
		return errors.CombineErrors(err, readable.Close())
	}
	defer func() { _ = r.Close() }()
	layout, err := r.Layout()
	if err != nil {
		return err
	}

	rh := objstorage.MakeNoopReadHandle(readable)
	br := r.BlockReader()

	block := func(kind block.Kind, handle block.Handle) {
		if err == nil {
			err = fa.sstBlock(ctx, &rh, kind, handle, br)
		}
	}
	for i := range layout.Data {
		block(blockkind.SSTableData, layout.Data[i].Handle)
	}
	for i := range layout.Index {
		block(blockkind.SSTableIndex, layout.Index[i])
	}
	block(blockkind.SSTableIndex, layout.TopIndex)
	for i := range layout.Filter {
		block(blockkind.Filter, layout.Filter[i].Handle)
	}
	block(blockkind.RangeDel, layout.RangeDel)
	block(blockkind.RangeKey, layout.RangeKey)
	for i := range layout.ValueBlock {
		block(blockkind.SSTableValue, layout.ValueBlock[i])
	}
	block(blockkind.Metadata, layout.ValueIndex)
	return err
}

func (fa *FileAnalyzer) sstBlock(
	ctx context.Context, rh objstorage.ReadHandle, kind block.Kind, bh block.Handle, r *block.Reader,
) error {
	if bh.Length == 0 {
		return nil
	}
	if fa.readLimiter != nil {
		if err := fa.readLimiter.WaitCtx(ctx, tokenbucket.Tokens(bh.Length)); err != nil {
			return err
		}
	}
	h, err := r.Read(ctx, block.NoReadEnv, rh, bh, kind, func(*block.Metadata, []byte) error { return nil })
	if err != nil {
		return err
	}
	fa.BlockAnalyzer.Block(kind, h.BlockData())
	h.Release()
	return nil
}
