ALTER TABLE questions
ADD COLUMN region_scope TEXT NOT NULL DEFAULT 'general'
CHECK(region_scope IN ('domestic', 'general', 'overseas'));

CREATE INDEX idx_questions_set_region ON questions(set_id, region_scope);
