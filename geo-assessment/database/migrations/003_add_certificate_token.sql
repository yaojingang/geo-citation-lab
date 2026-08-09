ALTER TABLE attempts
ADD COLUMN certificate_token TEXT;

CREATE UNIQUE INDEX idx_attempts_certificate_token
ON attempts(certificate_token)
WHERE certificate_token IS NOT NULL;
