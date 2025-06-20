// Copyright 2025 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package block

import (
	"math/rand/v2"
	"testing"

	"github.com/cockroachdb/pebble/internal/compression"
	"github.com/cockroachdb/pebble/sstable/block/blockkind"
	"github.com/stretchr/testify/require"
)

func TestCompressor(t *testing.T) {
	settings := []compression.Setting{
		compression.None,
		compression.Snappy,
		compression.MinLZFastest,
		compression.ZstdLevel3,
	}

	src := make([]byte, 1024)
	dst := make([]byte, 0, 1024)
	for runs := 0; runs < 100; runs++ {
		profile := &CompressionProfile{
			DataBlocks:          settings[rand.IntN(len(settings))],
			ValueBlocks:         settings[rand.IntN(len(settings))],
			OtherBlocks:         settings[rand.IntN(len(settings))],
			MinReductionPercent: 0,
		}

		compressor := MakeCompressor(profile)
		ci, _ := compressor.Compress(dst, src, blockkind.SSTableData)
		require.Equal(t, compressionIndicatorFromAlgorithm(profile.DataBlocks.Algorithm), ci)

		ci, _ = compressor.Compress(dst, src, blockkind.SSTableValue)
		require.Equal(t, compressionIndicatorFromAlgorithm(profile.ValueBlocks.Algorithm), ci)

		ci, _ = compressor.Compress(dst, src, blockkind.BlobValue)
		require.Equal(t, compressionIndicatorFromAlgorithm(profile.ValueBlocks.Algorithm), ci)

		ci, _ = compressor.Compress(dst, src, blockkind.Index)
		require.Equal(t, compressionIndicatorFromAlgorithm(profile.OtherBlocks.Algorithm), ci)

		ci, _ = compressor.Compress(dst, src, blockkind.Metadata)
		require.Equal(t, compressionIndicatorFromAlgorithm(profile.OtherBlocks.Algorithm), ci)

		compressor.Close()
	}
}
