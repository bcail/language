Value lang_sqlite3_version(void) {
  const char* version = sqlite3_libversion();
  Value s = OBJ_VAL(copy_string(version, (uint32_t) strlen(version)));
  inc_ref(AS_OBJ(s));
  return s;
}


Value lang_sqlite3_open(Value file_name) {
  sqlite3* db;
  sqlite3_open(AS_CSTRING(file_name), &db);
  return SQLITE3_VAL(db);
}


Value lang_sqlite3_close(Value db) {
  sqlite3_close(AS_SQLITE3(db));
  return NIL_VAL;
}
