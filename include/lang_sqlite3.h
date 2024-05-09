Value lang_sqlite3_version(void) {
  const char* version = sqlite3_libversion();
  Value s = copy_string(version, (uint32_t) strlen(version));
  if (IS_OBJ(s)) {
    inc_ref(AS_OBJ(s));
  }
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

int process_row(void* results, int num_columns, char** result_strings, char** result_columns) {
  ObjList* row = allocate_list((uint32_t) num_columns);
  for (int i=0; i < num_columns; i++) {
    list_add(row, copy_string(result_strings[i], (uint32_t) strlen(result_strings[i])));
  }
  list_add(results, OBJ_VAL(row));
  inc_ref(row);
  return 0;
}

Value lang_sqlite3_execute(Value db, Value sql_code) {
  ObjList* results = allocate_list(0);
  char* error = NULL;
  int exec_result = sqlite3_exec(AS_SQLITE3(db), AS_CSTRING(sql_code), process_row, results, &error);
  if (exec_result != 0) {
    printf("error: %s", error);
    fflush(stdout);
    sqlite3_free(error);
  }
  if (results->count == 0) {
    FREE(ObjList, results);
    return NIL_VAL;
  } else {
    inc_ref(results);
    return OBJ_VAL(results);
  }
}
