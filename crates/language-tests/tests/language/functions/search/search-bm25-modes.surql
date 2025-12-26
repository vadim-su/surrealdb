/**
[test]
reason = "Compare BM25 (fast) and BM25_ACCURATE scoring modes"

[[test.results]]
value = "NONE"

[[test.results]]
value = "NONE"

[[test.results]]
value = "NONE"

[[test.results]]
value = "[{ content: 'Graph databases excel at traversal.', id: doc_fast:1 }]"

[[test.results]]
value = "[{ content: 'Relational databases use SQL.', id: doc_fast:2 }]"

[[test.results]]
value = "[{ content: 'This has no relevant terms.', id: doc_fast:3 }]"

[[test.results]]
value = "[{ content: 'Graph databases excel at traversal.', id: doc_accurate:1 }]"

[[test.results]]
value = "[{ content: 'Relational databases use SQL.', id: doc_accurate:2 }]"

[[test.results]]
value = "[{ content: 'This has no relevant terms.', id: doc_accurate:3 }]"

[[test.results]]
value = "[{ id: doc_fast:2, score: 0.4937678575515747f }, { id: doc_fast:1, score: 0.45895916223526f }]"

[[test.results]]
value = "[{ id: doc_accurate:2, score: 0.4937678575515747f }, { id: doc_accurate:1, score: 0.45895916223526f }]"

*/

-- Setup analyzer
DEFINE ANALYZER test TOKENIZERS blank, class, punct FILTERS lowercase;

-- BM25 (fast mode - uses SmallFloat-encoded lengths, ~12.5% precision)
DEFINE INDEX idx_fast ON TABLE doc_fast FIELDS content FULLTEXT ANALYZER test BM25;

-- BM25_ACCURATE (accurate mode - uses exact document lengths)
DEFINE INDEX idx_accurate ON TABLE doc_accurate FIELDS content FULLTEXT ANALYZER test BM25_ACCURATE;

-- Create test documents for fast mode
CREATE doc_fast:1 SET content = "Graph databases excel at traversal.";
CREATE doc_fast:2 SET content = "Relational databases use SQL.";
CREATE doc_fast:3 SET content = "This has no relevant terms.";

-- Create identical documents for accurate mode
CREATE doc_accurate:1 SET content = "Graph databases excel at traversal.";
CREATE doc_accurate:2 SET content = "Relational databases use SQL.";
CREATE doc_accurate:3 SET content = "This has no relevant terms.";

-- Query fast mode: documents get different scores based on SmallFloat-encoded length
-- (shorter doc:2 scores higher than longer doc:1 - same as accurate mode)
SELECT id, search::score(1) AS score FROM doc_fast WHERE content @1@ 'databases' ORDER BY score DESC;

-- Query accurate mode: documents get different scores based on exact length
-- (shorter doc:2 "Relational databases use SQL." scores higher than longer doc:1)
SELECT id, search::score(1) AS score FROM doc_accurate WHERE content @1@ 'databases' ORDER BY score DESC;
