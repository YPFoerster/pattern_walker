DROP TABLE IF EXISTS ptrp_crossrefs_params;
CREATE TABLE ptrp_crossrefs_params (string_len INT, gamma FLOAT, overlap FLOAT, number_refs INT);
INSERT INTO ptrp_crossrefs_params
SELECT DISTINCT string_len,gamma,overlap,number_refs from ptrp_crossrefs;
