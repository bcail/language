Value lang_sqlite3_version(void) {
  const char* version = sqlite3_libversion();
  Value s = OBJ_VAL(copy_string(version, (uint32_t) strlen(version)));
  inc_ref(AS_OBJ(s));
  return s;
}


Value lang_sqlite3_open(Value file_name) {
  sqlite3* db;
  sqlite3_open(AS_CSTRING(file_name), &db);
  ObjSqlite3* sqlite3_obj = ALLOCATE_OBJ(ObjSqlite3, OBJ_SQLITE3_DB);
  sqlite3_obj->db = db;
  inc_ref(sqlite3_obj);
  return OBJ_VAL(sqlite3_obj);
}


Value lang_sqlite3_close(Value db) {
  return NIL_VAL;
}
