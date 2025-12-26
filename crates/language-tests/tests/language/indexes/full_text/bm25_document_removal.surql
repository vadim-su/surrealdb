/**
[test]
reason = "Test that BM25 scoring updates correctly when documents are removed"

[[test.results]]
value = "NONE"

[[test.results]]
value = "NONE"

[[test.results]]
value = "[{ content: 'databases are powerful tools', id: removal_test:1 }]"

[[test.results]]
value = "[{ content: 'databases store information', id: removal_test:2 }]"

[[test.results]]
value = "[{ content: 'search engines use databases', id: removal_test:3 }]"

[[test.results]]
value = "[{ id: removal_test:2, score: 0.14426159858703613f }, { id: removal_test:1, score: 0.12874341011047363f }, { id: removal_test:3, score: 0.12874341011047363f }]"

[[test.results]]
value = "[]"

[[test.results]]
value = "[{ id: removal_test:1, score: 0.18232156336307526f }, { id: removal_test:3, score: 0.18232156336307526f }]"

[[test.results]]
value = "[]"

[[test.results]]
value = "[{ id: removal_test:3, score: 0.28768208622932434f }]"

*/

-- Setup analyzer
DEFINE ANALYZER removal_analyzer TOKENIZERS blank, class FILTERS lowercase;

-- Create BM25 index
DEFINE INDEX idx_removal ON TABLE removal_test FIELDS content FULLTEXT ANALYZER removal_analyzer BM25;

-- Create test documents
CREATE removal_test:1 SET content = "databases are powerful tools";
CREATE removal_test:2 SET content = "databases store information";
CREATE removal_test:3 SET content = "search engines use databases";

-- Query before deletion - all 3 docs should match "databases"
SELECT id, search::score(1) AS score FROM removal_test WHERE content @1@ 'databases' ORDER BY score DESC;

-- Delete one document
DELETE removal_test:2;

-- Query after first deletion - only 2 docs should match
SELECT id, search::score(1) AS score FROM removal_test WHERE content @1@ 'databases' ORDER BY score DESC;

-- Delete another document
DELETE removal_test:1;

-- Query after second deletion - only 1 doc should match
SELECT id, search::score(1) AS score FROM removal_test WHERE content @1@ 'databases' ORDER BY score DESC;
