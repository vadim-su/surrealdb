/**
[test]

[[test.results]]
value = "None"

[[test.results]]
value = "None"

[[test.results]]
value = "{ events: {  }, fields: {  }, indexes: { idx_accurate: 'DEFINE INDEX idx_accurate ON doc FIELDS content FULLTEXT ANALYZER test BM25_ACCURATE(1.5,0.8)' }, lives: {  }, tables: {  } }"

[[test.results]]
value = "None"

[[test.results]]
value = "{ events: {  }, fields: {  }, indexes: { idx_default: 'DEFINE INDEX idx_default ON doc2 FIELDS content FULLTEXT ANALYZER test BM25_ACCURATE(1.2,0.75)' }, lives: {  }, tables: {  } }"

*/

-- Define analyzer
DEFINE ANALYZER test TOKENIZERS blank;

-- BM25_ACCURATE with custom parameters
DEFINE INDEX idx_accurate ON TABLE doc FIELDS content FULLTEXT ANALYZER test BM25_ACCURATE(1.5, 0.8);

-- Verify the index definition
INFO FOR TABLE doc;

-- BM25_ACCURATE with default parameters
DEFINE INDEX idx_default ON TABLE doc2 FIELDS content FULLTEXT ANALYZER test BM25_ACCURATE;

-- Verify the default parameters
INFO FOR TABLE doc2;
