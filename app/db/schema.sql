-- Decision-Grade RAG — PostgreSQL Schema
-- psql -U postgres -d policy_rag -f schema.sql

CREATE TABLE IF NOT EXISTS policy_rates (
    id               SERIAL PRIMARY KEY,
    policy_code      VARCHAR(20)  NOT NULL,
    category         VARCHAR(50)  NOT NULL,
    grade            VARCHAR(20)  NOT NULL,
    travel_type      VARCHAR(20),
    per_day_inr      NUMERIC(12,2),
    per_night_inr    NUMERIC(12,2),
    annual_limit_inr NUMERIC(14,2),
    currency         CHAR(3)     DEFAULT 'INR',
    department       VARCHAR(100) DEFAULT 'ALL',
    effective_date   DATE,
    notes            TEXT,
    UNIQUE(policy_code, grade, category, travel_type)
);

CREATE INDEX IF NOT EXISTS idx_pr_grade    ON policy_rates(grade);
CREATE INDEX IF NOT EXISTS idx_pr_category ON policy_rates(category);
CREATE INDEX IF NOT EXISTS idx_pr_code     ON policy_rates(policy_code);

CREATE TABLE IF NOT EXISTS query_audit (
    id            BIGSERIAL PRIMARY KEY,
    query_text    TEXT,
    route         VARCHAR(20),
    grade         VARCHAR(20),
    verified      BOOLEAN,
    retries       INTEGER DEFAULT 0,
    latency_ms    INTEGER,
    queried_at    TIMESTAMP DEFAULT NOW()
);

-- Seed data
INSERT INTO policy_rates(policy_code,category,grade,travel_type,per_day_inr,per_night_inr,annual_limit_inr,effective_date,notes) VALUES
('T-04','meal','L1-L3','domestic',600,NULL,NULL,'2025-04-01','Meal per day'),
('T-04','meal','L4-L5','domestic',900,NULL,NULL,'2025-04-01','Meal per day'),
('T-04','meal','L6-L7','domestic',1200,NULL,NULL,'2025-04-01','Meal per day'),
('T-04','meal','VP','domestic',1800,NULL,NULL,'2025-04-01','Meal per day'),
('T-04','hotel','L1-L3','domestic',NULL,3500,NULL,'2025-04-01','Hotel per night'),
('T-04','hotel','L4-L5','domestic',NULL,5500,NULL,'2025-04-01','Hotel per night'),
('T-04','hotel','L6-L7','domestic',NULL,8000,NULL,'2025-04-01','Hotel per night'),
('T-04','hotel','VP','domestic',NULL,12000,NULL,'2025-04-01','Hotel per night'),
('T-04','per_diem','L1-L3','international',4150,9960,NULL,'2025-04-01','USD50/day USD120/night'),
('T-04','per_diem','L4-L5','international',6225,14940,NULL,'2025-04-01','USD75/day USD180/night'),
('T-04','per_diem','L6-L7','international',8300,20750,NULL,'2025-04-01','USD100/day USD250/night'),
('T-04','per_diem','VP','international',12450,33200,NULL,'2025-04-01','USD150/day USD400/night'),
('IT-09','laptop','L1-L3',NULL,NULL,NULL,55000,'2024-07-01','4yr cycle'),
('IT-09','laptop','L4-L5',NULL,NULL,NULL,85000,'2024-07-01','3yr cycle'),
('IT-09','laptop','L6-L7',NULL,NULL,NULL,120000,'2024-07-01','3yr cycle'),
('IT-09','laptop','VP',NULL,NULL,NULL,180000,'2024-07-01','Premium+dock')
ON CONFLICT DO NOTHING;
