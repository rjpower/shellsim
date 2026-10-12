.bail on
CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);
BEGIN;
INSERT INTO items VALUES (1, 'committed');
COMMIT;
BEGIN;
INSERT INTO items VALUES (2, 'rolled back');
ROLLBACK;
SELECT id, name FROM items ORDER BY id;
