/* Exercise the shared guest provider independently of CPython's extension. */
#include <assert.h>
#include <sqlite3.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    sqlite3 *db;
    sqlite3_stmt *statement;
    assert(sqlite3_open("/work/native.db", &db) == SQLITE_OK);
    assert(sqlite3_exec(db, "CREATE TABLE items (name TEXT); BEGIN", 0, 0, 0) == SQLITE_OK);
    assert(sqlite3_prepare_v2(db, "INSERT INTO items VALUES (?)", -1, &statement, 0) == SQLITE_OK);
    assert(sqlite3_bind_text(statement, 1, "committed ' value", -1, SQLITE_STATIC) == SQLITE_OK);
    assert(sqlite3_step(statement) == SQLITE_DONE);
    assert(sqlite3_finalize(statement) == SQLITE_OK);
    assert(sqlite3_exec(db, "COMMIT; BEGIN; INSERT INTO items VALUES ('discard'); ROLLBACK", 0, 0, 0) == SQLITE_OK);
    assert(sqlite3_close(db) == SQLITE_OK);
    assert(sqlite3_open("/work/native.db", &db) == SQLITE_OK);
    assert(sqlite3_prepare_v2(db, "SELECT name FROM items", -1, &statement, 0) == SQLITE_OK);
    assert(sqlite3_step(statement) == SQLITE_ROW);
    assert(strcmp((const char *)sqlite3_column_text(statement, 0), "committed ' value") == 0);
    assert(sqlite3_step(statement) == SQLITE_DONE);
    assert(sqlite3_finalize(statement) == SQLITE_OK);
    assert(sqlite3_close(db) == SQLITE_OK);
    puts("SQLite shared provider transactions passed");
    return 0;
}
