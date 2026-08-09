CREATE TABLE users (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX idx_sessions_user ON sessions(user_id);

CREATE TABLE question_sets (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    total_points INTEGER NOT NULL CHECK(total_points = 100),
    time_limit_seconds INTEGER NOT NULL CHECK(time_limit_seconds > 0),
    scoring_version TEXT NOT NULL,
    evidence_frozen_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_question_sets_one_active ON question_sets(active) WHERE active = 1;

CREATE TABLE questions (
    id TEXT PRIMARY KEY,
    set_id TEXT NOT NULL REFERENCES question_sets(id) ON DELETE CASCADE,
    code TEXT NOT NULL,
    title TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('single', 'multiple')),
    dimension TEXT NOT NULL,
    dimension_label TEXT NOT NULL,
    difficulty TEXT NOT NULL CHECK(difficulty IN ('basic', 'advanced', 'challenge')),
    difficulty_label TEXT NOT NULL,
    cognitive_level TEXT NOT NULL,
    prompt TEXT NOT NULL,
    weight INTEGER NOT NULL CHECK(weight IN (3, 4)),
    sort_order INTEGER NOT NULL,
    misconception_tag TEXT NOT NULL,
    explanation TEXT NOT NULL,
    principle TEXT NOT NULL,
    expected_seconds INTEGER NOT NULL CHECK(expected_seconds > 0 AND expected_seconds <= 180),
    source_refs_json TEXT NOT NULL,
    source_objects_json TEXT NOT NULL,
    UNIQUE(set_id, code)
);
CREATE INDEX idx_questions_set_order ON questions(set_id, sort_order);

CREATE TABLE choices (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    code TEXT NOT NULL,
    text TEXT NOT NULL,
    is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
    rationale TEXT NOT NULL,
    UNIQUE(question_id, code)
);
CREATE INDEX idx_choices_question ON choices(question_id, code);

CREATE TABLE attempts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    set_id TEXT NOT NULL REFERENCES question_sets(id),
    attempt_no INTEGER NOT NULL CHECK(attempt_no BETWEEN 1 AND 10),
    status TEXT NOT NULL CHECK(status IN ('in_progress', 'submitted', 'timed_out')),
    started_at TEXT NOT NULL,
    deadline_at TEXT NOT NULL,
    submitted_at TEXT,
    duration_seconds INTEGER CHECK(duration_seconds BETWEEN 0 AND 1800),
    score INTEGER CHECK(score BETWEEN 0 AND 100),
    correct_count INTEGER CHECK(correct_count BETWEEN 0 AND 30),
    scoring_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id, attempt_no)
);
CREATE UNIQUE INDEX idx_attempts_one_in_progress ON attempts(user_id) WHERE status = 'in_progress';
CREATE INDEX idx_attempts_user_history ON attempts(user_id, attempt_no DESC);
CREATE INDEX idx_attempts_cohort ON attempts(set_id, status, submitted_at);

CREATE TABLE attempt_items (
    attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
    question_id TEXT NOT NULL REFERENCES questions(id),
    position INTEGER NOT NULL CHECK(position BETWEEN 1 AND 30),
    question_snapshot_json TEXT NOT NULL,
    choice_order_json TEXT NOT NULL,
    response_json TEXT NOT NULL DEFAULT '[]',
    points INTEGER CHECK(points >= 0),
    time_spent_seconds INTEGER NOT NULL DEFAULT 0 CHECK(time_spent_seconds BETWEEN 0 AND 1800),
    activity_seq INTEGER NOT NULL DEFAULT 0 CHECK(activity_seq >= 0),
    change_count INTEGER NOT NULL DEFAULT 0 CHECK(change_count >= 0),
    first_viewed_at TEXT,
    last_viewed_at TEXT,
    PRIMARY KEY(attempt_id, question_id),
    UNIQUE(attempt_id, position)
);
CREATE INDEX idx_attempt_items_attempt_position ON attempt_items(attempt_id, position);

CREATE TABLE rate_limits (
    key_hash TEXT NOT NULL,
    action TEXT NOT NULL,
    window_started_at TEXT NOT NULL,
    count INTEGER NOT NULL CHECK(count >= 0),
    expires_at TEXT NOT NULL,
    PRIMARY KEY(key_hash, action, window_started_at)
);
CREATE INDEX idx_rate_limits_expires ON rate_limits(expires_at);
