DROP TABLE IF EXISTS no_seed_params;
CREATE TABLE no_seed_params (string_len INT, gamma FLOAT, overlap FLOAT);
INSERT INTO no_seed_params
SELECT DISTINCT string_len,gamma,overlap from no_seed;
