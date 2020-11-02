DROP TABLE IF EXISTS redraw_patterns_params;
CREATE TABLE redraw_patterns_params (string_len INT, gamma FLOAT, overlap FLOAT);
INSERT INTO redraw_patterns_params
SELECT DISTINCT string_len,gamma,overlap from redraw_patterns;
