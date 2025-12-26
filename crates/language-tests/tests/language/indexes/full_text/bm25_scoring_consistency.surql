/**
[test]
reason = "Test that BM25 and BM25_ACCURATE produce consistent ranking"

[[test.results]]
value = "NONE"

[[test.results]]
value = "NONE"

[[test.results]]
value = "NONE"

[[test.results]]
value = "[{ content: 'short', id: consistency_fast:1 }]"

[[test.results]]
value = "[{ content: 'medium length document', id: consistency_fast:2 }]"

[[test.results]]
value = "[{ content: 'this is a much longer document with more words', id: consistency_fast:3 }]"

[[test.results]]
value = "[{ content: 'short', id: consistency_accurate:1 }]"

[[test.results]]
value = "[{ content: 'medium length document', id: consistency_accurate:2 }]"

[[test.results]]
value = "[{ content: 'this is a much longer document with more words', id: consistency_accurate:3 }]"

[[test.results]]
error = false

[[test.results]]
error = false

*/

-- Setup analyzer
DEFINE ANALYZER consistency_analyzer TOKENIZERS blank, class FILTERS lowercase;

-- Create BM25 (fast) index
DEFINE INDEX idx_fast ON TABLE consistency_fast FIELDS content FULLTEXT ANALYZER consistency_analyzer BM25;

-- Create BM25_ACCURATE index
DEFINE INDEX idx_accurate ON TABLE consistency_accurate FIELDS content FULLTEXT ANALYZER consistency_analyzer BM25_ACCURATE;

-- Create identical documents for BM25 (fast)
CREATE consistency_fast:1 SET content = "short";
CREATE consistency_fast:2 SET content = "medium length document";
CREATE consistency_fast:3 SET content = "this is a much longer document with more words";

-- Create identical documents for BM25_ACCURATE
CREATE consistency_accurate:1 SET content = "short";
CREATE consistency_accurate:2 SET content = "medium length document";
CREATE consistency_accurate:3 SET content = "this is a much longer document with more words";

-- Query BM25 (fast) - should rank doc:2 highest (has "document" with good length)
SELECT id, search::score(1) AS score FROM consistency_fast WHERE content @1@ 'document' ORDER BY score DESC;

-- Query BM25_ACCURATE - should have same ranking as fast
SELECT id, search::score(1) AS score FROM consistency_accurate WHERE content @1@ 'document' ORDER BY score DESC;
