import copy
from enum import Enum, auto
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile



############################################
# LANG C CODE
############################################

INCLUDES = '''
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
'''

LANG_C_CODE = '''
#define ALLOCATE(type, count) \
    (type*)reallocate(NULL, sizeof(type) * (count))

#define ALLOCATE_OBJ(type, objectType) \
    (type*)allocateObject(sizeof(type), objectType)

#define GROW_CAPACITY(capacity) \
            ((capacity) < 8 ? 8 : (capacity) * 2)

#define GROW_ARRAY(type, pointer, oldCount, newCount) \
            (type*)reallocate(pointer, sizeof(type) * (newCount))

#define FREE(type, pointer) reallocate(pointer, (size_t)0)

#define FREE_ARRAY(type, pointer) \
            reallocate(pointer, (size_t)0)

#define NIL_VAL  ((Value){NIL, {.boolean = 0}})
#define TOMBSTONE_VAL  ((Value){TOMBSTONE, {.boolean = 0}})
#define BOOL_VAL(value)  ((Value){BOOL, {.boolean = value}})
#define NUMBER_VAL(value)  ((Value){NUMBER, {.number = value}})
#define RECUR_VAL(value)  ((Value){RECUR, {.recur = value}})
#define FILE_VAL(value)   ((Value){FILE_HANDLE, {.file = (FILE*)value}})
#define SHORT_STRING_VAL(value)  ((Value){SHORT_STRING, {.short_string = value}})
#define OBJ_VAL(object)   ((Value){OBJ, {.obj = (Obj*)object}})
#define AS_BOOL(value)  ((value).data.boolean)
#define AS_NUMBER(value)  ((value).data.number)
#define AS_RATIO(value)  ((value).data.ratio)
#define AS_RECUR(value)       ((value).data.recur)
#define AS_FILE(value)       ((value).data.file)
#define AS_SHORT_STRING(value)  ((value).data.short_string)
#define AS_SHORT_CSTRING(value)  ((value).data.short_string.chars)
#define AS_OBJ(value)  ((value).data.obj)
#define AS_STRING(value)       ((ObjString*)AS_OBJ(value))
#define AS_CSTRING(value)      (((ObjString*)AS_OBJ(value))->chars)
#define AS_LIST(value)       ((ObjList*)AS_OBJ(value))
#define AS_MAP(value)       ((ObjMap*)AS_OBJ(value))
#define IS_NIL(value)  ((value).type == NIL)
#define IS_TOMBSTONE(value)  ((value).type == TOMBSTONE)
#define IS_BOOL(value)  ((value).type == BOOL)
#define IS_NUMBER(value)  ((value).type == NUMBER)
#define IS_RATIO(value)  ((value).type == RATIO)
#define IS_RECUR(value)  ((value).type == RECUR)
#define IS_ERROR(value)  ((value).type == ERROR)
#define IS_FILE(value)  ((value).type == FILE_HANDLE)
#define IS_SHORT_STRING(value)  ((value).type == SHORT_STRING)
#define IS_OBJ(value)  ((value).type == OBJ)
#define IS_STRING(value)  isObjType(value, OBJ_STRING)
#define IS_LIST(value)  isObjType(value, OBJ_LIST)
#define IS_MAP(value)  isObjType(value, OBJ_MAP)
#if defined(USE_SQLITE3)
  #define SQLITE3_VAL(value)   ((Value){SQLITE3_DB, {.db = (sqlite3*)value}})
  #define IS_SQLITE3(value)  ((value).type == SQLITE3_DB)
  #define AS_SQLITE3(value)      ((value).data.db)
#endif
#define FLOAT_EQUAL_THRESHOLD 1e-7
#define MAP_EMPTY (-1)
#define MAP_TOMBSTONE (-2)
#define MAP_MAX_LOAD 0.75
#define MAX_LINE 1000
#define ERROR_GENERAL 0
#define ERROR_TYPE 1
#define ERROR_OUT_OF_BOUNDS 2
#define ERROR_DIVIDE_BY_ZERO 3

void* reallocate(void* pointer, size_t newSize) {
  if (newSize == 0) {
    free(pointer);
    return NULL;
  }

  void* result = realloc(pointer, newSize);
  return result;
}

typedef enum {
  OBJ_STRING,
  OBJ_LIST,
  OBJ_MAP,
} ObjType;

typedef struct {
  ObjType type;
  uint32_t ref_cnt;
} Obj;

typedef enum {
  NIL,
  BOOL,
  NUMBER,
  RATIO,
  RECUR,
  TOMBSTONE,
  ERROR,
  FILE_HANDLE,
  SHORT_STRING,
  OBJ,
#if defined(USE_SQLITE3)
  SQLITE3_DB,
#endif
} ValueType;

typedef struct {
  uint8_t type;
  char message[7];
} ErrorInfo;

typedef struct {
  uint8_t length;
  char chars[7];
} ShortString;

typedef struct {
  int32_t numerator;
  int32_t denominator;
} Ratio;

typedef struct Recur Recur;

typedef struct {
  ValueType type;
  union {
    bool boolean;
    double number;
    ErrorInfo err_info;
    Ratio ratio;
    Obj* obj;
    Recur* recur;
    FILE* file;
    ShortString short_string;
#if defined(USE_SQLITE3)
    sqlite3* db;
#endif
  } data;
} Value;

Value error_val(uint8_t type, char* message) {
  char buf[7];
  strncpy(buf, message, 6);
  buf[6] = 0;
  ErrorInfo info;
  info.type = type;
  strncpy(info.message, buf, 7);
  Value v = {ERROR, {.err_info = info}};
  return v;
}

int32_t integer_gcd(int32_t a, int32_t b) {
  if (a < 0) {
    a = a * -1;
  }
  if (b < 0) {
    b = b * -1;
  }
  if (a == b) {
    return a;
  }
  if (a == 0) {
    return b;
  }
  if (b == 0) {
    return a;
  }
  // a must be greater than b
  if (b > a) {
    int32_t c = a;
    a = b;
    b = c;
  }
  // a - b, b
  while (true) {
    a = a - b;
    if (a == b) {
      return a;
    }
    if (a < b) {
      return a;
    }
  }
}

Value ratio_val(int32_t numer, int32_t denom) {
  Ratio ratio;
  int32_t gcd = integer_gcd(numer, denom);
  if (gcd > 1) {
    ratio.numerator = numer / gcd;
    ratio.denominator = denom / gcd;
  } else {
    ratio.numerator = numer;
    ratio.denominator = denom;
  }
  Value v = {RATIO, {.ratio = ratio}};
  return v;
}

static inline bool isObjType(Value value, ObjType type) {
  return IS_OBJ(value) && AS_OBJ(value)->type == type;
}

typedef struct {
  Obj obj;
  uint32_t length;
  uint32_t hash;
  char* chars;
} ObjString;

struct Recur {
  uint32_t count;
  uint32_t capacity;
  Value* values;
};

typedef struct {
  Obj obj;
  uint32_t count;
  uint32_t capacity;
  Value* values;
} ObjList;

/* Maps
 * Ideas from:
 *   https://github.com/python/cpython/blob/main/Objects/dictobject.c
 *   https://github.com/python/cpython/blob/main/Include/internal/pycore_dict.h
 *   https://mail.python.org/pipermail/python-dev/2012-December/123028.html
 *   https://morepypy.blogspot.com/2015/01/faster-more-memory-efficient-and-more.html
 * Use a sparse list that contains indices into a compact list of MapEntries
 *
 * MAP_EMPTY (-1) - marks a slot as never used
 * 0 - 2147483648 - marks an index into the entries list
 *
 * MinSize - starting size of new dict - 8 might be good
 */

typedef struct {
  Value key;
  Value value;
} MapEntry;

typedef struct {
  Obj obj;
  uint32_t num_entries;
  uint32_t indices_capacity;
  uint32_t entries_capacity;
  int32_t* indices; /* start with always using int32 for now */
  MapEntry* entries;
} ObjMap;

static Obj* allocateObject(size_t size, ObjType type) {
  Obj* object = (Obj*)reallocate(NULL, size);
  object->type = type;
  object->ref_cnt = 0;
  return object;
}

void free_object(Obj* object);

// http://www.toccata.io/2019/02/RefCounting.html
void inc_ref(Obj* object) {
  object->ref_cnt++;
}

void dec_ref_and_free(Obj* object) {
  object->ref_cnt--;
  if (object->ref_cnt == 0) {
    free_object(object);
  }
}

static uint32_t hash_number(double number) {
  uint32_t hash = 2166136261u;

  char prefix = 'n';
  hash ^= (uint8_t) prefix;
  hash *= 16777619;

  char str[100];
  int32_t num_chars = sprintf(str, "%g", number);

  for (int32_t i = 0; i < num_chars; i++) {
    hash ^= (uint8_t) str[i];
    hash *= 16777619;
  }
  return hash;
}

static uint32_t hash_short_string(ShortString sh) {
  uint32_t hash = 2166136261u;
  char prefix = 'x';
  hash ^= (uint8_t) prefix;
  hash *= 16777619;

  for (uint32_t i = 0; i < sh.length; i++) {
    hash ^= (uint8_t) sh.chars[i];
    hash *= 16777619;
  }
  return hash;
}

static uint32_t hash_string(const char* key, uint32_t length) {
  uint32_t hash = 2166136261u;

  char prefix = 's';
  hash ^= (uint8_t) prefix;
  hash *= 16777619;

  for (uint32_t i = 0; i < length; i++) {
    hash ^= (uint8_t)key[i];
    hash *= 16777619;
  }
  return hash;
}

uint32_t _hash(Value v) {
  if (IS_NIL(v)) {
    uint32_t hash = 2166136261u;
    hash ^= (uint8_t) 0;
    hash *= 16777619;
    return hash;
  }
  else if (IS_BOOL(v)) {
    uint32_t hash = 2166136261u;
    if (AS_BOOL(v) == false) {
      hash ^= (uint8_t) 1;
    } else {
      hash ^= (uint8_t) 2;
    }
    hash *= 16777619;
    return hash;
  }
  else if (IS_NUMBER(v)) {
    return hash_number(AS_NUMBER(v));
  }
  else if (IS_SHORT_STRING(v)) {
    return hash_short_string(AS_SHORT_STRING(v));
  }
  else if (IS_STRING(v)) {
    ObjString* s = AS_STRING(v);
    return s->hash;
  }
  else {
    return 0;
  }
}

Value hash(Value v) {
  return NUMBER_VAL((double) (_hash(v)));
}

Value map_set(ObjMap* map, Value key, Value value);

static Value allocate_string(char* chars, uint32_t length, uint32_t hash) {
  ObjString* string = ALLOCATE_OBJ(ObjString, OBJ_STRING);
  string->length = length;
  string->hash = hash;
  string->chars = chars;
  return OBJ_VAL(string);
}

Value copy_string(const char* chars, uint32_t length) {
  if (length < 7) {
    ShortString sh;
    sh.length = (uint8_t) length;
    strcpy(sh.chars, chars);
    return SHORT_STRING_VAL(sh);
  }
  uint32_t hash = hash_string(chars, length);
  char* heapChars = ALLOCATE(char, length + 1);
  memcpy(heapChars, chars, (size_t)length);
  heapChars[length] = 0; /* terminate it w/ NULL, so we can pass c-string to functions that need it */
  return allocate_string(heapChars, length, hash);
}

ObjList* allocate_list(uint32_t initial_capacity) {
  ObjList* list = ALLOCATE_OBJ(ObjList, OBJ_LIST);
  list->count = 0;
  list->capacity = initial_capacity;
  if (initial_capacity == 0) {
    list->values = NULL;
  } else {
    list->values = GROW_ARRAY(Value, NULL, 0, (size_t) initial_capacity);
  }
  return list;
}

void list_add(ObjList* list, Value item) {
  if (list->capacity < list->count + 1) {
    uint32_t oldCapacity = list->capacity;
    list->capacity = GROW_CAPACITY(oldCapacity);
    list->values = GROW_ARRAY(Value, list->values, oldCapacity, list->capacity);
  }

  list->values[list->count] = item;
  list->count++;
  if (IS_OBJ(item)) {
    inc_ref(AS_OBJ(item));
  }
}

Value list_conj(Value lst, Value item) {
  if (IS_LIST(lst)) {
    list_add(AS_LIST(lst), item);
    return lst;
  }
  return error_val(ERROR_TYPE, "");
}

Value list_count(Value list) {
  return NUMBER_VAL((double) AS_LIST(list)->count);
}

Value count(Value v) {
  if (IS_SHORT_STRING(v)) {
    return NUMBER_VAL((double) AS_SHORT_STRING(v).length);
  } else if (IS_STRING(v)) {
    return NUMBER_VAL((double) AS_STRING(v)->length);
  } else if (IS_LIST(v)) {
    return list_count(v);
  }
  return error_val(ERROR_TYPE, "");
}

Value list_get(Value list, int32_t index) {
  if (index < 0) {
    return NIL_VAL;
  }

  if ((uint32_t) index < AS_LIST(list)->count) {
    return AS_LIST(list)->values[index];
  }
  else {
    return NIL_VAL;
  }
}

Value list_remove(Value list, Value index) {
  ObjList* obj_list = AS_LIST(list);
  if (AS_NUMBER(index) < 0 || (uint32_t) AS_NUMBER(index) > obj_list->count) {
    return NIL_VAL;
  }
  uint32_t i = (uint32_t) AS_NUMBER(index);
  while (i < obj_list->count) {
    if ((i+1) == obj_list->count) {
      obj_list->values[i] = NIL_VAL;
    } else {
      obj_list->values[i] = obj_list->values[i+1];
    }
    i++;
  }
  obj_list->count--;
  return list;
}

void swap(Value v[], uint32_t i, uint32_t j) {
  if (i == j) {
    return;
  }
  Value temp = v[i];
  v[i] = v[j];
  v[j] = temp;
}

Value nil_Q_(Value value) {
  return BOOL_VAL(IS_NIL(value));
}

bool double_equal(double x, double y) {
  double diff = fabs(x - y);
  return diff < FLOAT_EQUAL_THRESHOLD;
}

Value add_two_ratios(Ratio x, Ratio y) {
  if (x.denominator == y.denominator) {
    int32_t numerator = x.numerator + y.numerator;
    return ratio_val(numerator, x.denominator);
  } else {
    int32_t numerator = (x.numerator * y.denominator) + (y.numerator * x.denominator);
    int32_t denominator = x.denominator * y.denominator;
    return ratio_val(numerator, denominator);
  }
}

Value add_two(Value x, Value y) {
  if (IS_NUMBER(x) && IS_NUMBER(y)) {
    return NUMBER_VAL(AS_NUMBER(x) + AS_NUMBER(y));
  }
  if (IS_RATIO(x) && IS_RATIO(y)) {
    return add_two_ratios(AS_RATIO(x), AS_RATIO(y));
  }
  return error_val(ERROR_TYPE, "");
}

Value add_list(Value numbers) {
  ObjList* numbers_list = AS_LIST(numbers);
  Value item = numbers_list->values[0];
  if (IS_NUMBER(item)) {
    double result = AS_NUMBER(item);
    for (uint32_t i = 1; i < numbers_list->count; i++) {
      item = numbers_list->values[i];
      if (!IS_NUMBER(item)) {
        return error_val(ERROR_TYPE, "");
      }
      result += AS_NUMBER(item);
    }
    return NUMBER_VAL(result);
  } else if (IS_RATIO(item)) {
    Ratio result = AS_RATIO(item);
    for (uint32_t i = 1; i < numbers_list->count; i++) {
      item = numbers_list->values[i];
      if (!IS_RATIO(item)) {
        return error_val(ERROR_TYPE, "");
      }
      result = AS_RATIO(add_two_ratios(result, AS_RATIO(item)));
    }
    return ratio_val(result.numerator, result.denominator);
  }
  return error_val(ERROR_TYPE, "");
}

Value subtract_two(Value x, Value y) {
  if (!IS_NUMBER(x) || !IS_NUMBER(y)) {
    return error_val(ERROR_TYPE, "");
  }
  return NUMBER_VAL(AS_NUMBER(x) - AS_NUMBER(y));
}

Value subtract_list(Value numbers) {
  ObjList* numbers_list = AS_LIST(numbers);
  Value item = numbers_list->values[0];
  if (!IS_NUMBER(item)) {
    return error_val(ERROR_TYPE, "");
  }
  double result = AS_NUMBER(item);
  for (uint32_t i = 1; i < numbers_list->count; i++) {
    item = numbers_list->values[i];
    if (!IS_NUMBER(item)) {
      return error_val(ERROR_TYPE, "");
    }
    result = result - AS_NUMBER(item);
  }
  return NUMBER_VAL(result);
}

Value multiply_two(Value x, Value y) {
  if (!IS_NUMBER(x) || !IS_NUMBER(y)) {
    return error_val(ERROR_TYPE, "");
  }
  return NUMBER_VAL(AS_NUMBER(x) * AS_NUMBER(y));
}

Value multiply_list(Value numbers) {
  ObjList* numbers_list = AS_LIST(numbers);
  Value item = numbers_list->values[0];
  if (!IS_NUMBER(item)) {
    return error_val(ERROR_TYPE, "");
  }
  double result = AS_NUMBER(item);
  for (uint32_t i = 1; i < numbers_list->count; i++) {
    item = numbers_list->values[i];
    if (!IS_NUMBER(item)) {
      return error_val(ERROR_TYPE, "");
    }
    result = result * AS_NUMBER(item);
  }
  return NUMBER_VAL(result);
}

Value divide_two(Value x, Value y) {
  if (!IS_NUMBER(x) || !IS_NUMBER(y)) {
    return error_val(ERROR_TYPE, "");
  }
  if (double_equal(AS_NUMBER(y), 0)) {
    return error_val(ERROR_DIVIDE_BY_ZERO, "");
  }
  return NUMBER_VAL(AS_NUMBER(x) / AS_NUMBER(y));
}

Value divide_list(Value numbers) {
  ObjList* numbers_list = AS_LIST(numbers);
  Value item = numbers_list->values[0];
  if (!IS_NUMBER(item)) {
    return error_val(ERROR_TYPE, "");
  }
  double result = AS_NUMBER(item);
  for (uint32_t i = 1; i < numbers_list->count; i++) {
    item = numbers_list->values[i];
    if (!IS_NUMBER(item)) {
      return error_val(ERROR_TYPE, "");
    }
    if (double_equal(AS_NUMBER(item), 0)) {
      return error_val(ERROR_DIVIDE_BY_ZERO, "");
    }
    result = result / AS_NUMBER(item);
  }
  return NUMBER_VAL(result);
}

Value greater(ObjMap* user_globals, Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) > AS_NUMBER(y)); }
Value greater_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) >= AS_NUMBER(y)); }
Value less_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) <= AS_NUMBER(y)); }
Value less(ObjMap* user_globals, Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) < AS_NUMBER(y)); }

Value to_number(Value v) {
  if (IS_SHORT_STRING(v)) {
    const char *str = AS_SHORT_CSTRING(v);
    char* ptr;
    long int value = strtol(str, &ptr, 10);
    return NUMBER_VAL((double) value);
  } else if (IS_STRING(v)) {
    const char *str = AS_CSTRING(v);
    char* ptr;
    long int value = strtol(str, &ptr, 10);
    return NUMBER_VAL((double) value);
  } else if (IS_NUMBER(v)) {
    return v;
  }
  return error_val(ERROR_TYPE, "");
}

void quick_sort(ObjMap* user_globals, Value v[], uint32_t left, uint32_t right, Value (*compare) (ObjMap*, Value, Value)) {
  /* C Programming Language K&R p87*/
  uint32_t i, last;
  if (left >= right) {
    return;
  }
  if ((int) left < 0) {
    return;
  }
  if ((int) right < 0) {
    return;
  }
  swap(v, left, (left + right)/2);
  last = left;
  for (i = left+1; i <= right; i++) {
    if (AS_BOOL((*compare) (user_globals, v[i], v[left]))) {
      swap(v, ++last, i);
    }
  }
  swap(v, left, last);
  quick_sort(user_globals, v, left, last-1, *compare);
  quick_sort(user_globals, v, last+1, right, *compare);
}

Value list_sort(ObjMap* user_globals, Value list, Value (*compare) (ObjMap*, Value, Value)) {
  ObjList* lst = AS_LIST(list);
  quick_sort(user_globals, lst->values, 0, (lst->count)-1, *compare);
  return OBJ_VAL(lst);
}

void recur_init(Recur* recur) {
  recur->count = 0;
  recur->capacity = 0;
  recur->values = NULL;
}

void recur_free(Recur* recur) {
  for (uint32_t i = 0; i < recur->count; i++) {
    Value v = recur->values[i];
    if (IS_OBJ(v)) {
      dec_ref_and_free(AS_OBJ(v));
    }
  }
  FREE_ARRAY(Value, recur->values);
  recur_init(recur);
}

void recur_add(Recur* recur, Value item) {
  if (recur->capacity < recur->count + 1) {
    uint32_t oldCapacity = recur->capacity;
    recur->capacity = GROW_CAPACITY(oldCapacity);
    recur->values = GROW_ARRAY(Value, recur->values, oldCapacity, recur->capacity);
  }

  recur->values[recur->count] = item;
  recur->count++;
  if (IS_OBJ(item)) {
    inc_ref(AS_OBJ(item));
  }
}

Value recur_get(Value recur, uint32_t index) {
  if (index < AS_RECUR(recur)->count) {
    return AS_RECUR(recur)->values[index];
  }
  else {
    return NIL_VAL;
  }
}

ObjMap* allocate_map(void) {
  ObjMap* map = ALLOCATE_OBJ(ObjMap, OBJ_MAP);
  map->num_entries = 0;
  map->indices_capacity = 0;
  map->entries_capacity = 0;
  map->indices = NULL;
  map->entries = NULL;
  return map;
}

Value map_count(Value map) {
  return NUMBER_VAL((double) AS_MAP(map)->num_entries);
}

bool is_truthy(Value value) {
  if (IS_NIL(value)) {
    return false;
  }
  if (IS_BOOL(value)) {
    if (AS_BOOL(value) == false) {
      return false;
    }
  }
  return true;
}

Value equal(Value x, Value y) {
  if (x.type != y.type) {
    return BOOL_VAL(false);
  }
  else if (IS_NIL(x)) {
    return BOOL_VAL(true);
  }
  else if (IS_BOOL(x)) {
    return BOOL_VAL(AS_BOOL(x) == AS_BOOL(y));
  }
  else if (IS_NUMBER(x)) {
    return BOOL_VAL(double_equal(AS_NUMBER(x), AS_NUMBER(y)));
  }
  else if (IS_SHORT_STRING(x)) {
    ShortString xShort = AS_SHORT_STRING(x);
    ShortString yShort = AS_SHORT_STRING(y);
    if ((xShort.length == yShort.length) &&
        (memcmp(xShort.chars, yShort.chars, (size_t)xShort.length) == 0)) {
      return BOOL_VAL(true);
    }
    return BOOL_VAL(false);
  }
  else if (IS_STRING(x)) {
    ObjString* xString = AS_STRING(x);
    ObjString* yString = AS_STRING(y);
    if ((xString->length == yString->length) &&
        (memcmp(xString->chars, yString->chars, (size_t)xString->length) == 0)) {
      return BOOL_VAL(true);
    }
    return BOOL_VAL(false);
  }
  else if (IS_LIST(x)) {
    ObjList* xList = AS_LIST(x);
    ObjList* yList = AS_LIST(y);
    if (xList->count == yList->count) {
      Value num_items = list_count(x);
      for (int32_t i = 0; i < AS_NUMBER(num_items); i++) {
        Value xItem = list_get(x, i);
        Value yItem = list_get(y, i);
        if (!AS_BOOL(equal(xItem, yItem))) {
          return BOOL_VAL(false);
        }
      }
      return BOOL_VAL(true);
    }
    return BOOL_VAL(false);
  }
  else if (IS_MAP(x)) {
    ObjMap* xMap = AS_MAP(x);
    ObjMap* yMap = AS_MAP(y);
    uint32_t x_num_items = xMap->num_entries;
    uint32_t y_num_items = yMap->num_entries;
    if (x_num_items != y_num_items) {
      return BOOL_VAL(false);
    }
    for (uint32_t i = 0; i < xMap->num_entries; i++) {
      MapEntry x_entry = xMap->entries[i];
      MapEntry y_entry = yMap->entries[i];
      if (!AS_BOOL(equal(x_entry.key, y_entry.key))) {
        return BOOL_VAL(false);
      }
      if (!AS_BOOL(equal(x_entry.value, y_entry.value))) {
        return BOOL_VAL(false);
      }
    }
    return BOOL_VAL(true);
  }
  else {
    return BOOL_VAL(false);
  }
}

static int32_t find_indices_index(int32_t* indices, MapEntry* entries, uint32_t capacity, Value key) {
  /* hash the key and get an index
   * - if indices[index] is empty, return it
   * - if indices[index] points to an entry in entries with a hash that matches our hash, return index
   * Otherwise, keep adding one till we get to the correct key or an empty slot. */

  uint32_t index = _hash(key) % capacity;
  for (;;) {
    if (indices[index] == MAP_EMPTY) {
      return (int32_t) index;
    }
    if (AS_BOOL(equal(key, entries[indices[index]].key))) {
      return (int32_t) index;
    }

    index = (index + 1) % capacity;
  }
}

static void adjustCapacity(ObjMap* map, uint32_t capacity) {
  // allocate new space
  int32_t* indices = ALLOCATE(int32_t, capacity);
  MapEntry* entries = ALLOCATE(MapEntry, capacity);

  // initialize all indices to MAP_EMPTY
  for (uint32_t i = 0; i < capacity; i++) {
    indices[i] = MAP_EMPTY;
  }

  // copy entries over to new space, filling in indices slots as well
  uint32_t num_entries = map->num_entries;
  uint32_t entries_index = 0;
  for (; entries_index < num_entries; entries_index++) {
    // find new index
    int32_t indices_index = find_indices_index(indices, entries, capacity, map->entries[entries_index].key);
    indices[indices_index] = (int32_t) entries_index;
    entries[entries_index] = map->entries[entries_index];
  }

  // fill in remaining entries with nil values
  for (; entries_index < capacity; entries_index++) {
    entries[entries_index].key = NIL_VAL;
    entries[entries_index].value = NIL_VAL;
  }

  FREE_ARRAY(int32_t, map->indices);
  FREE_ARRAY(MapEntry, map->entries);

  map->indices_capacity = capacity;
  map->entries_capacity = capacity;
  map->indices = indices;
  map->entries = entries;
}

Value map_set(ObjMap* map, Value key, Value value) {
  /* keep indices & entries same number of entries for now */
  if ((double)map->num_entries + 1 > (double)map->indices_capacity * MAP_MAX_LOAD) {
    uint32_t capacity = GROW_CAPACITY(map->indices_capacity);
    adjustCapacity(map, capacity);
  }

  int32_t indices_index = find_indices_index(map->indices, map->entries, map->indices_capacity, key);
  int32_t entries_index = (int32_t) map->indices[indices_index];

  bool isNewKey = (entries_index == MAP_EMPTY);
  if (isNewKey) {
    entries_index = (int32_t) map->num_entries;
    map->num_entries++;
    map->indices[indices_index] = entries_index;
  }

  if (IS_OBJ(key)) {
    inc_ref(AS_OBJ(key));
  }
  if (IS_OBJ(value)) {
    inc_ref(AS_OBJ(value));
  }

  MapEntry* entry = &(map->entries[entries_index]);

  if (IS_OBJ(entry->key)) {
    dec_ref_and_free(AS_OBJ(entry->key));
  }
  if (IS_OBJ(entry->value)) {
    dec_ref_and_free(AS_OBJ(entry->value));
  }

  entry->key = key;
  entry->value = value;
  return OBJ_VAL(map);
}

Value map_remove(Value map, Value key) {
  ObjMap* obj_map = AS_MAP(map);

  if (obj_map->num_entries == 0) {
    return map;
  }

  int32_t indices_index = find_indices_index(obj_map->indices, obj_map->entries, obj_map->indices_capacity, key);
  int32_t entries_index = (int32_t) obj_map->indices[indices_index];

  bool isNewKey = (entries_index == MAP_EMPTY);
  if (isNewKey) {
    return map;
  }

  obj_map->indices[indices_index] = MAP_TOMBSTONE;

  MapEntry* entry = &(obj_map->entries[entries_index]);

  if (IS_OBJ(entry->key)) {
    dec_ref_and_free(AS_OBJ(entry->key));
  }
  if (IS_OBJ(entry->value)) {
    dec_ref_and_free(AS_OBJ(entry->value));
  }

  entry->key = TOMBSTONE_VAL;
  entry->value = TOMBSTONE_VAL;

  obj_map->num_entries--;
  return map;
}

Value map_contains(ObjMap* map, Value key) {
  if (map->num_entries == 0) {
    return BOOL_VAL(false);
  }
  int32_t indices_index = find_indices_index(map->indices, map->entries, map->indices_capacity, key);
  int32_t entries_index = map->indices[indices_index];
  bool isNewKey = (entries_index == MAP_EMPTY);
  if (isNewKey) {
    return BOOL_VAL(false);
  }
  else {
    return BOOL_VAL(true);
  }
}

Value map_get(ObjMap* map, Value key, Value defaultVal) {
  if (map->num_entries == 0) {
    return defaultVal;
  }

  int32_t indices_index = find_indices_index(map->indices, map->entries, map->indices_capacity, key);
  int32_t entries_index = map->indices[indices_index];

  bool isNewKey = (entries_index == MAP_EMPTY);
  if (isNewKey) {
    return defaultVal;
  }
  else {
    return map->entries[entries_index].value;
  }
}

Value map_keys(ObjMap* map) {
  uint32_t num_entries = map->num_entries;
  ObjList* keys = allocate_list((uint32_t) num_entries);
  for (uint32_t i = 0; i < num_entries; i++) {
    list_add(keys, map->entries[i].key);
  }
  return OBJ_VAL(keys);
}

Value map_vals(ObjMap* map) {
  uint32_t num_entries = map->num_entries;
  ObjList* vals = allocate_list((uint32_t) num_entries);
  for (uint32_t i = 0; i < num_entries; i++) {
    list_add(vals, map->entries[i].value);
  }
  return OBJ_VAL(vals);
}

Value map_pairs(ObjMap* map) {
  uint32_t num_entries = map->num_entries;
  ObjList* pairs = allocate_list((uint32_t) num_entries);
  for (uint32_t i = 0; i < num_entries; i++) {
    if (!AS_BOOL(equal(map->entries[i].key, NIL_VAL))) {
      ObjList* pair = allocate_list((uint32_t) 2);
      list_add(pair, map->entries[i].key);
      list_add(pair, map->entries[i].value);
      list_add(pairs, OBJ_VAL(pair));
    }
  }
  return OBJ_VAL(pairs);
}

bool is_integer(double n) {
  if (ceil(n) == n) {
    return true;
  }
  return false;
}

Value print(Value value) {
  if IS_NIL(value) {
    printf("nil");
  }
  else if IS_BOOL(value) {
    if AS_BOOL(value) {
      printf("true");
    }
    else {
      printf("false");
    }
  }
  else if IS_NUMBER(value) {
    double n = AS_NUMBER(value);
    if (is_integer(n)) {
      printf("%.f", n);
    } else {
      printf("%g", n);
    }
  }
  else if IS_RATIO(value) {
    Ratio r = AS_RATIO(value);
    printf("%d", r.numerator);
    printf("/");
    printf("%d", r.denominator);
  }
  else if (IS_ERROR(value)) {
    if (value.data.err_info.type == ERROR_DIVIDE_BY_ZERO) {
      printf("ERROR: DivideByZero");
    } else if (value.data.err_info.type == ERROR_OUT_OF_BOUNDS) {
      printf("ERROR: OutOfBounds");
    } else if (value.data.err_info.type == ERROR_TYPE) {
      printf("ERROR: Type");
    } else {
      printf("ERROR: General");
    }
  }
  else if (IS_LIST(value)) {
    Value num_items = list_count(value);
    printf("[");
    if (AS_NUMBER(num_items) > 0) {
      print(list_get(value, 0));
    }
    for (int i = 1; i < AS_NUMBER(num_items); i++) {
      printf(" ");
      print(list_get(value, i));
    }
    printf("]");
  }
  else if (IS_MAP(value)) {
    ObjMap* map = AS_MAP(value);
    uint32_t num_entries = map->num_entries;
    printf("{");
    bool first_entry = true;
    for (uint32_t i = 0; i < num_entries; i++) {
      if (IS_TOMBSTONE(map->entries[i].key)) {
        num_entries++;
        continue;
      }
      if (!first_entry) {
        printf(", ");
      }
      print(map->entries[i].key);
      printf(" ");
      print(map->entries[i].value);
      first_entry = false;
    }
    printf("}");
  }
  else if (IS_SHORT_STRING(value)) {
    printf("%s", AS_SHORT_CSTRING(value));
  }
  else {
    printf("%s", AS_CSTRING(value));
  }
  return NIL_VAL;
}

Value println(Value value) {
  print(value);
  printf("\\n");
  return NIL_VAL;
}

ObjList* cli_args;

Value readline(void) {
  /* K&R p29 */
  int ch = 0;
  char buffer[MAX_LINE];
  uint32_t num_chars;
  for (num_chars=0; num_chars<(MAX_LINE-1) && (ch=getchar()) != EOF && ch != '\\n'; num_chars++) {
    buffer[num_chars] = (char) ch;
  }
  if ((ch == EOF) && (num_chars == 0)) {
    return NIL_VAL;
  }
  if (num_chars < 7) {
    buffer[num_chars] = 0;
  }
  Value result = copy_string(buffer, num_chars);
  if (IS_OBJ(result)) {
    inc_ref(AS_OBJ(result));
  }
  return result;
}

Value str_blank(Value string) {
  if (IS_NIL(string)) {
    return BOOL_VAL(true);
  }
  if (IS_SHORT_STRING(string)) {
    ShortString s = AS_SHORT_STRING(string);
    if (s.length == 0) {
      return BOOL_VAL(true);
    }
    for (int i = 0; s.chars[i] != '\\0'; i++) {
      if (!isspace(s.chars[i])) {
        return BOOL_VAL(false);
      }
    }
    return BOOL_VAL(true);
  }
  ObjString* s = AS_STRING(string);
  if (s->length == 0) {
    return BOOL_VAL(true);
  }
  for (int i = 0; s->chars[i] != '\\0'; i++) {
    if (!isspace(s->chars[i])) {
      return BOOL_VAL(false);
    }
  }
  return BOOL_VAL(true);
}

Value str_lower(Value string) {
  if (IS_SHORT_STRING(string)) {
    ShortString s = AS_SHORT_STRING(string);
    char buffer[7];
    int i = 0;
    for (; s.chars[i] != 0; i++) {
      buffer[i] = (char) tolower((int) s.chars[i]);
    }
    buffer[i] = 0;
    ShortString s_lower;
    s_lower.length = s.length;
    memcpy(s_lower.chars, buffer, (size_t) (s.length+1));
    return SHORT_STRING_VAL(s_lower);
  } else {
    ObjString* s = AS_STRING(string);
    Value s_lower_val = copy_string(s->chars, s->length);
    ObjString* s_lower = AS_STRING(s_lower_val);
    for (int i=0; s_lower->chars[i] != '\\0'; i++) {
      s_lower->chars[i] = (char) tolower((int) s_lower->chars[i]);
    }
    return s_lower_val;
  }
}

Value str_split(Value string) {
  ObjList* splits = allocate_list((uint32_t) 0);
  uint32_t split_length = 0;
  int split_start_index = 0;

  if (IS_SHORT_STRING(string)) {
    ShortString s = AS_SHORT_STRING(string);
    for (int i=0; s.chars[i] != 0; i++) {
      if (s.chars[i] == ' ') {
        char split_buffer[7];
        memcpy(split_buffer, &(s.chars[split_start_index]), split_length);
        split_buffer[split_length] = 0;
        Value split = copy_string(split_buffer, split_length);
        list_add(splits, split);
        split_start_index = i + 1;
        split_length = 0;
      }
      else {
        split_length++;
      }
    }
    Value split = copy_string(&(s.chars[split_start_index]), split_length);
    list_add(splits, split);
  }
  // regular string
  else {
    ObjString* s = AS_STRING(string);
    for (int i=0; s->chars[i] != '\\0'; i++) {
      if (s->chars[i] == ' ') {
        if (split_length < 7) {
          char split_buffer[7];
          memcpy(split_buffer, &(s->chars[split_start_index]), split_length);
          split_buffer[split_length] = 0;
          Value split = copy_string(split_buffer, split_length);
          list_add(splits, split);
        } else {
          Value split = copy_string(&(s->chars[split_start_index]), split_length);
          list_add(splits, split);
        }
        split_start_index = i + 1;
        split_length = 0;
      }
      else {
        split_length++;
      }
    }
    Value split = copy_string(&(s->chars[split_start_index]), split_length);
    list_add(splits, split);
  }
  return OBJ_VAL(splits);
}

Value str_str(Value v) {
  // if we got a string, no need to do any work, just return it
  if (IS_STRING(v)) {
    inc_ref(AS_OBJ(v));
    return v;
  }
  if (IS_SHORT_STRING(v)) {
    return v;
  }

  Value s;
  if (IS_BOOL(v)) {
    if (AS_BOOL(v)) {
      s = copy_string("true", 4);
    }
    else {
      s = copy_string("false", 5);
    }
  }
  else if (IS_NUMBER(v)) {
    char str[100];
    int32_t num_chars = sprintf(str, "%g", AS_NUMBER(v));
    s = copy_string(str, (uint32_t) num_chars);
  }
  else if (IS_LIST(v)) {
    s = copy_string("[]", 2);
  }
  else if (IS_MAP(v)) {
    s = copy_string("{}", 2);
  }
  else {
    s = copy_string("", 0);
  }
  if (IS_OBJ(s)) {
    inc_ref(AS_OBJ(s));
  }
  return s;
}

Value str_join(Value list_val) {
  ObjList* list = AS_LIST(list_val);

  uint32_t num_bytes = 0;

  for (uint32_t i = 0; i < list->count; i++) {
    Value v = list_get(list_val, (int32_t)i);
    Value v_str = str_str(v);
    if (IS_SHORT_STRING(v_str)) {
      num_bytes = num_bytes + AS_SHORT_STRING(v_str).length;
    } else {
      num_bytes = num_bytes + AS_STRING(v_str)->length;
      dec_ref_and_free(AS_OBJ(v_str));
    }
    if (IS_OBJ(v)) {
      dec_ref_and_free(AS_OBJ(v));
    }
  }
  if (num_bytes < 7) {
    // construct a short-string instead
    char buffer[7];
    char* start_char = buffer;
    uint32_t i = 0;
    for (; i < list->count; i++) {
      Value v = list_get(list_val, (int32_t)i);
      Value v_str = str_str(v);
      ShortString v_short_string = AS_SHORT_STRING(v_str);
      memcpy(start_char, v_short_string.chars, (size_t)v_short_string.length);
      start_char = start_char + v_short_string.length;
    }
    buffer[num_bytes] = '\\0';
    ShortString sh;
    sh.length = (uint8_t) num_bytes;
    memcpy(sh.chars, buffer, num_bytes + 1);
    return SHORT_STRING_VAL(sh);
  }

  char* heapChars = ALLOCATE(char, (size_t)(num_bytes+1));
  char* start_char = heapChars;

  for (uint32_t i = 0; i < list->count; i++) {
    Value v = list_get(list_val, (int32_t)i);
    Value v_str = str_str(v);
    if (IS_STRING(v_str)) {
      ObjString* s = AS_STRING(v_str);
      memcpy(start_char, s->chars, (size_t)s->length);
      start_char = start_char + s->length;
    } else {
      // it's a short string
      ShortString sh = AS_SHORT_STRING(v_str);
      memcpy(start_char, sh.chars, (size_t)sh.length);
      start_char = start_char + sh.length;
    }
  }
  heapChars[num_bytes] = 0;
  uint32_t hash = hash_string(heapChars, num_bytes);
  return allocate_string(heapChars, num_bytes, hash);
}

Value str_index_of(Value s, Value value, Value from_index) {
  uint32_t index = 0;
  if (IS_NUMBER(from_index)) {
    index = (uint32_t) AS_NUMBER(from_index);
  }
  if (IS_SHORT_STRING(s)) {
    ShortString str = AS_SHORT_STRING(s);
    if (IS_SHORT_STRING(value)) {
      ShortString value_str = AS_SHORT_STRING(value);
      if (value_str.length <= str.length) {
        for (; index < str.length; index++) {
          if (str.chars[index] == value_str.chars[0]) {
            bool found = true;
            for (uint8_t j = 1; j < value_str.length; j++) {
              if (str.chars[index+j] != value_str.chars[j]) {
                found = false;
                break;
              }
            }
            if (found) {
              return NUMBER_VAL( (double) index);
            }
          }
        }
      }
    } else if (IS_STRING(value)) {
      // strings are longer than short strings, so we don't need to check
    } else {
      return error_val(ERROR_TYPE, "");
    }
  } else if (IS_STRING(s)) {
    ObjString* str = AS_STRING(s);
    if (IS_SHORT_STRING(value)) {
      ShortString value_str = AS_SHORT_STRING(value);
      for (; index < str->length; index++) {
        if (str->chars[index] == value_str.chars[0]) {
          bool found = true;
          for (uint8_t j = 1; j < value_str.length; j++) {
            if (str->chars[index+j] != value_str.chars[j]) {
              found = false;
              break;
            }
          }
          if (found) {
            return NUMBER_VAL( (double) index);
          }
        }
      }
    } else if (IS_STRING(value)) {
      ObjString* value_str = AS_STRING(value);
      if (value_str->length <= str->length) {
        for (; index < str->length; index++) {
          if (str->chars[index] == value_str->chars[0]) {
            bool found = true;
            for (uint32_t j = 1; j < value_str->length; j++) {
              if (str->chars[index+j] != value_str->chars[j]) {
                found = false;
                break;
              }
            }
            if (found) {
              return NUMBER_VAL( (double) index);
            }
          }
        }
      }
    } else {
      return error_val(ERROR_TYPE, "");
    }
  } else {
    return error_val(ERROR_TYPE, "");
  }
  return NIL_VAL;
}

Value str_subs(Value s, Value start, Value end) {
  if (IS_NUMBER(start)) {
    if (AS_NUMBER(start) < 0) {
      return error_val(ERROR_OUT_OF_BOUNDS, "");
    }
    uint32_t start_index = (uint32_t) AS_NUMBER(start);
    uint32_t num_bytes = 0;
    if (IS_SHORT_STRING(s)) {
      ShortString str = AS_SHORT_STRING(s);
      uint32_t end_index = 0;
      if (IS_NIL(end)) {
        end_index = (uint32_t) str.length;
      } else if (IS_NUMBER(end)) {
        if (AS_NUMBER(end) >= 0) {
          end_index = (uint32_t) AS_NUMBER(end);
        } else {
          return error_val(ERROR_OUT_OF_BOUNDS, "");
        }
      } else {
        return error_val(ERROR_TYPE, "");
      }
      if ((end_index < start_index) || (end_index > str.length)) {
        return error_val(ERROR_OUT_OF_BOUNDS, "");
      }
      char buffer[7];
      uint32_t i = start_index;
      uint32_t buf_index = 0;
      for (; i < end_index; i++) {
        buffer[buf_index] = str.chars[i];
        num_bytes = num_bytes + 1;
        buf_index += 1;
      }
      buffer[num_bytes] = 0;
      ShortString subs;
      subs.length = (uint8_t) num_bytes;
      memcpy(subs.chars, buffer, (size_t) (num_bytes+1));
      return SHORT_STRING_VAL(subs);
    } else if (IS_STRING(s)) {
      ObjString* str = AS_STRING(s);
      uint32_t end_index = 0;
      if (IS_NIL(end)) {
        end_index = (uint32_t) str->length;
      } else if (IS_NUMBER(end)) {
        if (AS_NUMBER(end) >= 0) {
          end_index = (uint32_t) AS_NUMBER(end);
        } else {
          return error_val(ERROR_OUT_OF_BOUNDS, "");
        }
      } else {
        return error_val(ERROR_TYPE, "");
      }
      if ((end_index < start_index) || (end_index > str->length)) {
        return error_val(ERROR_OUT_OF_BOUNDS, "");
      }
      // construct short string if it's a small results
      if ((end_index - start_index) < 7) {
        char buffer[7];
        uint32_t i = start_index;
        uint32_t buf_index = 0;
        for (; i < end_index; i++) {
          buffer[buf_index] = str->chars[i];
          num_bytes = num_bytes + 1;
          buf_index += 1;
        }
        buffer[num_bytes] = 0;
        ShortString subs;
        subs.length = (uint8_t) num_bytes;
        memcpy(subs.chars, buffer, (size_t) (num_bytes+1));
        return SHORT_STRING_VAL(subs);
      }
      // construct regular string
      else {
        size_t num_bytes_to_copy = end_index - start_index;
        char* heapChars = ALLOCATE(char, (size_t)(num_bytes_to_copy+1));
        char* start_char = heapChars;
        char* str_start_char = str->chars;
        str_start_char = str_start_char + start_index;
        memcpy(start_char, str_start_char, num_bytes_to_copy);
        heapChars[num_bytes_to_copy] = 0;
        uint32_t hash = hash_string(heapChars, num_bytes);
        return allocate_string(heapChars, num_bytes, hash);
      }
    }
  }
  return error_val(ERROR_TYPE, "");
}

Value math_gcd(Value param_1, Value param_2) {
  if (!IS_NUMBER(param_1) || !IS_NUMBER(param_2)) {
    return error_val(ERROR_TYPE, "");
  }
  double p1 = AS_NUMBER(param_1);
  double p2 = AS_NUMBER(param_2);
  if (!is_integer(p1) || !is_integer(p2)) {
    return error_val(ERROR_TYPE, "");
  }
  int32_t a = (int32_t) p1;
  int32_t b = (int32_t) p2;
  return NUMBER_VAL(integer_gcd(a, b));
}

Value file_open(Value path, const char* mode) {
  FILE* fp = fopen(AS_CSTRING(path), mode);
  return FILE_VAL(fp);
}

Value file_read(Value file) {
  int ch = 0;
  char buffer[MAX_LINE];
  uint32_t num_chars;
  FILE* fp = AS_FILE(file);
  for (num_chars=0; num_chars<(MAX_LINE-1) && (ch=getc(fp)) != EOF; num_chars++) {
    buffer[num_chars] = (char) ch;
  }
  if ((ch == EOF) && (num_chars == 0)) {
    return NIL_VAL;
  }
  if (num_chars < 7) {
    buffer[num_chars] = 0;
  }
  Value result = copy_string(buffer, num_chars);
  if (IS_OBJ(result)) {
    inc_ref(AS_OBJ(result));
  }
  return result;
}

Value file_write(Value file, Value data) {
  FILE* fp = AS_FILE(file);
  if (IS_SHORT_STRING(data)) {
    fprintf(fp, "%s", AS_SHORT_CSTRING(data));
  } else {
    fprintf(fp, "%s", AS_CSTRING(data));
  }
  fflush(fp);
  return NIL_VAL;
}

Value file_close(Value file) {
  fclose(AS_FILE(file));
  return NIL_VAL;
}

Value os_mkdir(Value dir_name) {
#if defined(WINDOWS)
  int result = _mkdir(AS_CSTRING(dir_name));
#else
  int result = mkdir(AS_CSTRING(dir_name), 0755);
#endif
  return NIL_VAL;
}

void free_object(Obj* object) {
  switch (object->type) {
    case OBJ_STRING: {
      ObjString* string = (ObjString*)object;
      FREE_ARRAY(char, string->chars);
      FREE(ObjString, object);
      break;
    }
    case OBJ_LIST: {
      ObjList* list = (ObjList*)object;
      for (uint32_t i = 0; i < list->count; i++) {
        Value v = list_get(OBJ_VAL(object), (int32_t)i);
        if (IS_OBJ(v)) {
          dec_ref_and_free(AS_OBJ(v));
        }
      }
      FREE_ARRAY(Value, list->values);
      FREE(ObjList, object);
      break;
    }
    case OBJ_MAP: {
      ObjMap* map = (ObjMap*)object;
      uint32_t num_entries = map->num_entries;
      for (uint32_t i = 0; i < num_entries; i++) {
        MapEntry entry = map->entries[i];
        if (IS_TOMBSTONE(entry.key)) {
          num_entries++;
          continue;
        }
        if (IS_OBJ(entry.key)) {
          dec_ref_and_free(AS_OBJ(entry.key));
        }
        if (IS_OBJ(entry.value)) {
          dec_ref_and_free(AS_OBJ(entry.value));
        }
      }
      FREE_ARRAY(int32_t, map->indices);
      FREE_ARRAY(MapEntry, map->entries);
      FREE(ObjMap, object);
      break;
    }
    default: {
      break;
    }
  }
}
'''

LANG_SQLITE_CODE = '''
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
'''

############################################
# COMPILER
############################################

RATIO_RE = re.compile(r'-?[0-9]+/-?[0-9]+')
NUMBER_RE = re.compile(r'-?[0-9]+\.?[0-9]*')


def _get_token(token_buffer):
    if RATIO_RE.match(token_buffer):
        return {'type': 'ratio', 'lexeme': token_buffer}
    elif NUMBER_RE.match(token_buffer):
        return {'type': 'number', 'lexeme': token_buffer}
    elif token_buffer == 'true':
        return {'type': 'true'}
    elif token_buffer == 'false':
        return {'type': 'false'}
    elif token_buffer == 'nil':
        return {'type': 'nil'}
    else:
        return {'type': 'symbol', 'lexeme': token_buffer}


def parse(source):
    ast = {'type': 'list', 'nodes': []}
    current_group = ast

    inside_string = False
    inside_comment = False
    token_buffer = ''

    for c in source:
        if inside_string:
            if c == '"':
                current_group['nodes'].append({'type': 'string', 'lexeme': token_buffer})
                token_buffer = ''
                inside_string = False
            else:
                token_buffer += c
        elif inside_comment:
            if c in ['\n', '\r']:
                inside_comment = False
        elif c in ['(', '[', '{']:
            if c == '[':
                group_type = 'vector'
            elif c == '{':
                group_type = 'map'
            else:
                group_type = 'list'
            new_group = {'type': group_type, 'nodes': []}
            current_group['nodes'].append(new_group)
            new_group['parent'] = current_group
            current_group = new_group
        elif c in [')', ']', '}']:
            if token_buffer:
                current_group['nodes'].append(_get_token(token_buffer))
                token_buffer = ''
            current_group = current_group['parent']
        elif c in [',', '\n', '\r']:
            pass
        elif c == ':':
            if token_buffer:
                raise Exception('invalid ":" char')
            token_buffer += c
        elif c.isalnum() or c in ['?', '.', '+', '-', '*', '/', '=', '>', '<']:
            token_buffer += c
        elif c == ' ':
            if token_buffer:
                current_group['nodes'].append(_get_token(token_buffer))
                token_buffer = ''
        elif c == '"':
            inside_string = True
        elif c == ';':
            inside_comment = True
        else:
            print(f'unhandled char "{c}"')

    if token_buffer:
        current_group['nodes'].append(_get_token(token_buffer))

    return ast


def nil_c(params, envs):
    param = compile_form(params[0], envs=envs)
    return f'nil_Q_({param["code"]})'


def add_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    name = _get_generated_name('add_result', envs=envs)
    envs[-1]['temps'].add(name)
    num_params = len(c_params)
    if num_params == 2:
        envs[-1]['code'].append(f'  Value {name} = add_two({c_params[0]}, {c_params[1]});')
    else:
        numbers_list_name = _get_generated_name('numbers', envs=envs)
        envs[-1]['temps'].add(numbers_list_name)
        envs[-1]['code'].append(f'  Value {numbers_list_name} = OBJ_VAL(allocate_list((uint32_t) {num_params}));\n  inc_ref(AS_OBJ({numbers_list_name}));')
        for param in c_params:
            envs[-1]['code'].append(f'  list_add(AS_LIST({numbers_list_name}), {param});')

        envs[-1]['code'].append(f'  Value {name} = add_list({numbers_list_name});')
        envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({numbers_list_name}));')
    return name


def subtract_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    name = _get_generated_name('subtract_result', envs=envs)
    envs[-1]['temps'].add(name)
    num_params = len(c_params)
    if num_params == 2:
        envs[-1]['code'].append(f'  Value {name} = subtract_two({c_params[0]}, {c_params[1]});')
    else:
        numbers_list_name = _get_generated_name('numbers', envs=envs)
        envs[-1]['temps'].add(numbers_list_name)
        envs[-1]['code'].append(f'  Value {numbers_list_name} = OBJ_VAL(allocate_list((uint32_t) {num_params}));\n  inc_ref(AS_OBJ({numbers_list_name}));')
        for param in c_params:
            envs[-1]['code'].append(f'  list_add(AS_LIST({numbers_list_name}), {param});')

        envs[-1]['code'].append(f'  Value {name} = subtract_list({numbers_list_name});')
        envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({numbers_list_name}));')
    return name


def multiply_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    name = _get_generated_name('multiply_result', envs=envs)
    envs[-1]['temps'].add(name)
    num_params = len(c_params)
    if num_params == 2:
        envs[-1]['code'].append(f'  Value {name} = multiply_two({c_params[0]}, {c_params[1]});')
    else:
        numbers_list_name = _get_generated_name('numbers', envs=envs)
        envs[-1]['temps'].add(numbers_list_name)
        envs[-1]['code'].append(f'  Value {numbers_list_name} = OBJ_VAL(allocate_list((uint32_t) {num_params}));\n  inc_ref(AS_OBJ({numbers_list_name}));')
        for param in c_params:
            envs[-1]['code'].append(f'  list_add(AS_LIST({numbers_list_name}), {param});')

        envs[-1]['code'].append(f'  Value {name} = multiply_list({numbers_list_name});')
        envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({numbers_list_name}));')
    return name


def divide_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    name = _get_generated_name('divide_result', envs=envs)
    envs[-1]['temps'].add(name)
    num_params = len(c_params)
    if num_params == 2:
        envs[-1]['code'].append(f'  Value {name} = divide_two({c_params[0]}, {c_params[1]});')
    else:
        numbers_list_name = _get_generated_name('numbers', envs=envs)
        envs[-1]['temps'].add(numbers_list_name)
        envs[-1]['code'].append(f'  Value {numbers_list_name} = OBJ_VAL(allocate_list((uint32_t) {num_params}));\n  inc_ref(AS_OBJ({numbers_list_name}));')
        for param in c_params:
            envs[-1]['code'].append(f'  list_add(AS_LIST({numbers_list_name}), {param});')

        envs[-1]['code'].append(f'  Value {name} = divide_list({numbers_list_name});')
        envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({numbers_list_name}));')
    return name


def equal_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return f'equal({c_params[0]}, {c_params[1]})'


def greater_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return f'greater(user_globals, {c_params[0]}, {c_params[1]})'


def greater_equal_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return f'greater_equal({c_params[0]}, {c_params[1]})'


def less_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return f'less(user_globals, {c_params[0]}, {c_params[1]})'


def less_equal_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return f'less_equal({c_params[0]}, {c_params[1]})'


def to_number_c(params, envs):
    param = compile_form(params[0], envs=envs)['code']
    return f'to_number({param})'


def hash_c(params, envs):
    result = compile_form(params[0], envs=envs)
    hash_result = _get_generated_name('hash_result', envs=envs)
    envs[-1]['temps'].add(hash_result)
    envs[-1]['code'].append(f'  Value {hash_result} = hash({result["code"]});')
    return hash_result


def def_c(params, envs):
    name = params[0]['lexeme']
    local_env = {'temps': envs[-1]['temps'], 'code': [], 'post': [], 'bindings': {}}
    envs.append(local_env)
    result = compile_form(params[1], envs=envs)
    c_name = _get_generated_name(base=f'u_{name}', envs=envs)
    current_ns = envs[0]['current_ns']
    envs[0]['namespaces'][current_ns][name] = {'type': 'var', 'c_name': c_name, 'code': result['code']}
    if local_env['code']:
        envs[0]['init'].extend(local_env['code'])
    if local_env['post']:
        envs[0]['post'].extend(local_env['post'])
    envs.pop()
    return ''


def _get_previous_bindings(envs):
    bindings = []
    for e in envs[1:]:
        for name, value in e.get('bindings', {}).items():
            if value and 'c_name' in value:
                bindings.append(value['c_name'])
            else:
                bindings.append(name)
    return list(set(bindings))


def if_form_c(params, envs):
    local_env = {'temps': envs[-1]['temps'], 'code': [], 'post': [], 'bindings': {}}
    envs.append(local_env)

    result_name = _get_generated_name('if_result', envs=envs)
    local_env['temps'].add(result_name)

    test_code = compile_form(params[0], envs=envs)['code']
    true_env = {'temps': local_env['temps'], 'code': [], 'post': [], 'bindings': {}}
    envs.append(true_env)
    true_result = compile_form(params[1], envs=envs)

    true_code = '\n  if (is_truthy(%s)) {\n  ' % test_code
    if true_env['code']:
        true_code += '\n  '.join(true_env['code']) + '\n  '
    if true_result['type'] == 'list' and true_result['nodes'][0]['type'] == 'symbol' and true_result['nodes'][0]['lexeme'] == 'recur':
        recur_name = envs[0]['recur_points'].pop()
        for r in true_result['nodes'][1:]:
            true_code += f'  recur_add(AS_RECUR({recur_name}), {r["code"]});\n'
        true_code += f'    {result_name} = {recur_name};'
    else:
        true_val = true_result['code']
        true_code += f'    {result_name} = {true_val};'
        # inc-ref result_name if needed, so object doesn't get freed
        if true_val not in ['BOOL_VAL(true)', 'BOOL_VAL(false)']:
            true_code += '\n    if (IS_OBJ(%s)) {\n      inc_ref(AS_OBJ(%s));\n    }' % (result_name, result_name)
    if true_env['post']:
        true_code += '\n  ' + '\n  '.join(true_env['post'])
    true_code += '\n  } // end true code\n'
    envs.pop()

    false_code = ''
    if len(params) > 2:
        false_env = {'temps': local_env['temps'], 'code': [], 'post': [], 'bindings': {}}
        envs.append(false_env)
        false_code += '  else {'
        false_result = compile_form(params[2], envs=envs)
        if false_env['code']:
            false_code += '\n' + '\n  '.join(false_env['code'])
        if false_result['type'] == 'list' and false_result['nodes'][0]['type'] == 'symbol' and false_result['nodes'][0]['lexeme'] == 'recur':
            recur_name = envs[0]['recur_points'].pop()
            for r in false_result['nodes'][1:]:
                false_code += f'\n    recur_add(AS_RECUR({recur_name}), {r["code"]});'
            false_code += f'\n    {result_name} = {recur_name};'
        else:
            false_val = false_result['code']
            false_code += f'\n  {result_name} = {false_val};'
            # inc-ref result_name if needed, so object doesn't get freed early
            if false_val not in ['BOOL_VAL(true)', 'BOOL_VAL(false)']:
                false_code += '\n    if (IS_OBJ(%s)) {\n      inc_ref(AS_OBJ(%s));\n    }' % (result_name, result_name)

        if false_env['post']:
            false_code += '\n' + '\n  '.join(false_env['post'])
        false_code += '\n  } // end false code'
        envs.pop()

    f_code = f'  Value {result_name} = NIL_VAL;'
    if local_env['code']:
        f_code += '\n' + '\n  '.join(local_env['code'])
    f_code += true_code
    f_code += false_code

    if local_env['post']:
        envs[-2]['post'].extend(local_env['post'])

    envs.pop()
    envs[-1]['code'].append(f_code)
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))

    return result_name


def let_c(params, envs):
    bindings = params[0]['nodes']
    body = params[1:]

    paired_bindings = []
    for i in range(0, len(bindings), 2):
        paired_bindings.append(bindings[i:i+2])

    f_params = 'ObjMap* user_globals'
    f_args = 'user_globals'

    previous_bindings = _get_previous_bindings(envs)
    if previous_bindings:
        for previous_binding in previous_bindings:
            f_params += f', Value {previous_binding}'
            f_args += f', {previous_binding}'

    local_env = {'temps': envs[-1]['temps'], 'code': [], 'post': [], 'bindings': {}}
    envs.append(local_env)
    for binding in paired_bindings:
        result = compile_form(binding[1], envs=envs)
        binding_name = _get_generated_name(base=binding[0]['lexeme'], envs=envs)
        result['c_name'] = binding_name
        local_env['bindings'][binding[0]['lexeme']] = result
        local_env['code'].append(f'  Value {binding_name} = {result["code"]};\n')

    f_code = ''

    expr_results = [compile_form(form, envs=envs) for form in body]
    final_result = expr_results[-1]

    if local_env['code']:
        f_code += '\n'.join(local_env['code']) + '\n'

    return_val = ''
    if final_result['type'] == 'list' and final_result['nodes'][0]['type'] == 'symbol' and final_result['nodes'][0]['lexeme'] == 'recur':
        recur_name = envs[0]['recur_points'].pop()
        for r in final_result['nodes'][1:]:
            f_code += f'\n    recur_add(AS_RECUR({recur_name}), {r["code"]});'
        return_val = recur_name
    else:
        return_val = final_result['code']
        if return_val not in ['NIL_VAL', 'BOOL_VAL(true)', 'BOOL_VAL(false)']:
            f_code += '\n  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }\n' % (return_val, return_val)

    if local_env['post']:
        f_code += '\n'.join(local_env['post']) + '\n'

    f_code += f'  return {return_val};'

    f_name = _get_generated_name(base='let', envs=envs)
    envs[0]['functions'][f_name] = 'Value %s(%s) {\n  %s\n}' % (f_name, f_params, f_code)

    result_name = _get_generated_name('let_result', envs=envs)
    envs[-1]['temps'].add(result_name)

    envs.pop()

    envs[-1]['code'].append(f'  Value {result_name} = {f_name}({f_args});')
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))

    return result_name


def _has_recur(expr):
    if expr['type'] == 'list':
        for e in expr['nodes']:
            if _has_recur(e):
                return True
    else:
        if expr['type'] == 'symbol' and expr['lexeme'] == 'recur':
            return True
    return False


def _loop(envs, bindings, exprs):
    loop_result = _get_generated_name('result', envs=envs)
    envs[-1]['temps'].add(loop_result)

    has_recur = _has_recur(exprs[-1])

    loop_post = []

    loop_code = ''

    if has_recur:
        recur_name = _get_generated_name('recur', envs=envs)
        envs[0]['recur_points'].append(recur_name)
        envs[-1]['temps'].add(recur_name)
        envs[-1]['bindings'][recur_name] = None

        loop_code += f'  Recur {recur_name}_1;'
        loop_code += f'\n  recur_init(&{recur_name}_1);'
        loop_code += f'\n  Value {recur_name} = RECUR_VAL(&{recur_name}_1);'

        loop_code += '\n  bool continueFlag = false;'
        loop_code += '\n  do {'

    for form in exprs[:-1]:
        form_env = {'temps': envs[-1]['temps'], 'code': [], 'post': [], 'bindings': {}}
        envs.append(form_env)
        compiled = compile_form(form, envs=envs)
        if form_env['code']:
            loop_code += '\n' + '\n  '.join(form_env['code'])
        if form_env['post']:
            loop_post.extend(form_env['post'])
        envs.pop()

    form_env = {'temps': envs[-1]['temps'], 'code': [], 'post': [], 'bindings': {}}
    envs.append(form_env)
    compiled = compile_form(exprs[-1], envs=envs)
    if form_env['code']:
        loop_code += '\n' + '\n  '.join(form_env['code'])

    loop_code += f'\n    Value {loop_result} = {compiled["code"]};'
    loop_code += '\n    if (IS_OBJ(%s)) {\n      inc_ref(AS_OBJ(%s));\n    }' % (loop_result, loop_result)

    if form_env['post']:
        loop_code += '\n' + '\n  '.join(form_env['post'])

    envs.pop()

    if has_recur:
        loop_code +=  '\n    if (IS_RECUR(%s)) {' % loop_result
        loop_code += f'\n      /* grab values from result and update  */'

        for index, var in enumerate(bindings):
            loop_code += '\n      if (IS_OBJ(%s)) {\n      dec_ref_and_free(AS_OBJ(%s));\n    }' % (var, var)
            loop_code += f'\n      {var} = recur_get({loop_result}, {index});'
            loop_code += '\n      if (IS_OBJ(%s)) {\n      inc_ref(AS_OBJ(%s));\n    }' % (var, var)

        if loop_post:
            loop_code += '\n' + '\n'.join(loop_post)

        loop_code += f'\n    continueFlag = true;'
        loop_code += f'\n    recur_free(&{recur_name}_1);'
        loop_code +=  '\n  }\n    else {\n'

        if loop_post:
            loop_code += '\n  '.join(loop_post)

        loop_code += f'\n      recur_free(&{recur_name}_1);'
        loop_code +=  '\n      return %s;\n    }' % loop_result

        loop_code += '\n  } while (continueFlag);'

    else:
        if loop_post:
            loop_code += '\n  '.join(loop_post)
        loop_code += '\n      return %s;' % loop_result

    loop_code += '\n  return NIL_VAL;'

    return loop_code


def loop_c(params, envs):
    bindings = params[0]['nodes']
    exprs = params[1:]

    loop_params = bindings[::2]
    initial_args = bindings[1::2]

    previous_bindings = _get_previous_bindings(envs)

    local_env = {'temps': set(previous_bindings).union(envs[-1]['temps']), 'code': [], 'post': [], 'bindings': {}}
    envs.append(local_env)

    f_code =  ''

    for index, loop_param in enumerate(loop_params):
        param_env = {'temps': envs[-1]['temps'], 'code': [], 'post': [], 'bindings': {}}
        envs.append(param_env)
        c_name = _get_generated_name(base=loop_param['lexeme'], envs=envs)
        result = compile_form(initial_args[index], envs=envs)
        local_env['bindings'][loop_param['lexeme']] = {
            'code': result['code'],
            'c_name': c_name,
        }
        if param_env['code']:
            f_code += '\n' + '\n'.join(param_env['code'])
        f_code += f'\n  Value {c_name} = {result["code"]};'
        f_code += '\n  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }' % (c_name, c_name)
        if param_env['post']:
            f_code += '\n' + '\n  '.join(param_env['post'])
        envs.pop()
        envs[-1]['temps'] = param_env['temps']

    loop_code = _loop(envs, [v['c_name'] for v in list(local_env['bindings'].values())], exprs)

    f_code += '\n' + loop_code

    c_loop_params = [f'Value {pb}' for pb in previous_bindings]
    c_loop_params_str = ', '.join(c_loop_params)
    if c_loop_params_str:
        c_loop_params_str = f'ObjMap* user_globals, {c_loop_params_str}'
    else:
        c_loop_params_str = 'ObjMap* user_globals'

    c_initial_args = ','.join([pb for pb in previous_bindings])
    if c_initial_args:
        c_initial_args = f'user_globals, {c_initial_args}'
    else:
        c_initial_args = 'user_globals'

    f_name = _get_generated_name(base='loop', envs=envs)

    envs[0]['functions'][f_name] = 'Value %s(%s) {\n%s\n}' % (f_name, c_loop_params_str, f_code)

    result_name = _get_generated_name('loop_result', envs=envs)

    envs.pop()

    envs[-1]['temps'].add(result_name)

    envs[-1]['code'].append(f'  Value {result_name} = {f_name}({c_initial_args});')
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))

    return result_name


def math_gcd_c(params, envs):
    param_1 = compile_form(params[0], envs=envs)['code']
    param_2 = compile_form(params[1], envs=envs)['code']
    name = _get_generated_name('math_gcd_result', envs=envs)
    envs[-1]['temps'].add(name)
    envs[-1]['code'].append(f'  Value {name} = math_gcd({param_1}, {param_2});')
    return name


def str_c(params, envs):
    if not params:
        arg_name = 'NIL_VAL'
    elif len(params) == 1:
        result = compile_form(params[0], envs=envs)
        arg_name = result['code']
    else:
        num_params = len(params)
        tmp_list_name = _get_generated_name('str_arg_tmp_list', envs=envs)
        name = _get_generated_name('str', envs=envs)
        envs[-1]['temps'].add(name)
        envs[-1]['code'].append(f'  Value {tmp_list_name} = OBJ_VAL(allocate_list((uint32_t) {num_params}));\n  inc_ref(AS_OBJ({tmp_list_name}));')
        for param in params:
            result = compile_form(param, envs=envs)
            envs[-1]['code'].append(f'  list_add(AS_LIST({tmp_list_name}), {result["code"]});')
        envs[-1]['code'].append(f'  Value {name} = str_join({tmp_list_name});\n')
        envs[-1]['code'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    inc_ref(AS_OBJ({name}));\n' + '  }')
        envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({tmp_list_name}));')
        envs[-1]['post'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    dec_ref_and_free(AS_OBJ({name}));\n' + '  }')
        return name

    name = _get_generated_name('str', envs=envs)
    envs[-1]['temps'].add(name)
    envs[-1]['code'].append(f'  Value {name} = str_str({arg_name});')
    # envs[-1]['code'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    dec_ref_and_free(AS_OBJ({name}));' + '\n  }')
    return name


def str_lower_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('str_lower_', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = str_lower({param_name});')
    envs[-1]['code'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    inc_ref(AS_OBJ({name}));' + '\n  }')
    envs[-1]['post'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    dec_ref_and_free(AS_OBJ({name}));' + '\n  }')
    return name


def str_blank_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('str_blank_', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = str_blank({param_name});')
    return name


def str_split_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('str_split_', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = str_split({param_name});')
    envs[-1]['code'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def str_index_of_c(params, envs):
    s_param = compile_form(params[0], envs=envs)['code']
    value_param = compile_form(params[1], envs=envs)['code']
    if len(params) > 2:
        from_index_param = compile_form(params[2], envs=envs)['code']
    else:
        from_index_param = compile_form({'type': 'number', 'lexeme': '0'}, envs=envs)['code']
    name = _get_generated_name('str_index_of_result', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = str_index_of({s_param}, {value_param}, {from_index_param});')
    return name


def str_subs_c(params, envs):
    s_param = compile_form(params[0], envs=envs)['code']
    start_param = compile_form(params[1], envs=envs)['code']
    if len(params) > 2:
        end_param = compile_form(params[2], envs=envs)['code']
    else:
        end_param = 'NIL_VAL'
    name = _get_generated_name('str_subs_result', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = str_subs({s_param}, {start_param}, {end_param});')
    envs[-1]['code'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    inc_ref(AS_OBJ({name}));' + '\n  }')
    envs[-1]['post'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    dec_ref_and_free(AS_OBJ({name}));' + '\n  }')
    return name


def nth_c(params, envs):
    lst = compile_form(params[0], envs=envs)['code']
    index = compile_form(params[1], envs=envs)['code']
    return f'list_get({lst}, (int32_t) AS_NUMBER({index}))'


def conj_c(params, envs):
    lst = compile_form(params[0], envs=envs)['code']
    item = compile_form(params[1], envs=envs)['code']
    return f'list_conj({lst}, {item})'


def remove_c(params, envs):
    lst = compile_form(params[0], envs=envs)
    index = compile_form(params[1], envs=envs)['code']
    return f'list_remove({lst["code"]}, {index})'


def sort_c(params, envs):
    if len(params) == 1:
        lst = compile_form(params[0], envs=envs)
        return f'list_sort(user_globals, {lst["code"]}, *less)'
    else:
        compare = compile_form(params[0], envs=envs)
        lst = compile_form(params[1], envs=envs)
        return f'list_sort(user_globals, {lst["code"]}, {compare["code"]})'


def count_c(params, envs):
    value = compile_form(params[0], envs=envs)['code']
    return f'count({value})'


def map_get_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    key = compile_form(params[1], envs=envs)['code']
    default = 'NIL_VAL'
    if len(params) > 2:
        default = compile_form(params[2], envs=envs)['code']
    name = _get_generated_name('map_get_', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = map_get(AS_MAP({m}), {key}, {default});')
    envs[-1]['code'].append('  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }' % (name, name))
    envs[-1]['temps'].add(name)
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (name, name))
    return name


def map_contains_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    key = compile_form(params[1], envs=envs)['code']
    return f'map_contains(AS_MAP({m}), {key})'


def map_assoc_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    key = compile_form(params[1], envs=envs)['code']
    value = compile_form(params[2], envs=envs)['code']
    result_name = _get_generated_name('map_assoc', envs=envs)
    envs[-1]['code'].append(f'  Value {result_name} = map_set(AS_MAP({m}), {key}, {value});')
    return result_name


def map_dissoc_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    key = compile_form(params[1], envs=envs)['code']
    result_name = _get_generated_name('map_dissoc', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['code'].append(f'  Value {result_name} = map_remove({m}, {key});')
    return result_name


def map_keys_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    name = _get_generated_name('map_keys_', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = map_keys(AS_MAP({m}));')
    envs[-1]['code'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def map_vals_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    name = _get_generated_name('map_vals_', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = map_vals(AS_MAP({m}));')
    envs[-1]['code'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def map_pairs_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    name = _get_generated_name('map_pairs_', envs=envs)
    envs[-1]['code'].append(f'  Value {name} = map_pairs(AS_MAP({m}));')
    envs[-1]['code'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def print_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('print_result', envs=envs)
    envs[-1]['temps'].add(name)
    envs[-1]['code'].append(f'  Value {name} = print({param_name});')
    # print always returns NIL_VAL, so don't need to dec_ref_and_free at the end
    return name


def println_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('println_result', envs=envs)
    envs[-1]['temps'].add(name)
    envs[-1]['code'].append(f'  Value {name} = println({param_name});')
    return name


def fn_c(params, envs, f_name=None):
    bindings = params[0]['nodes']
    exprs = params[1:]

    local_env = {'temps': envs[-1]['temps'], 'code': [], 'post': [], 'bindings': {}}
    envs.append(local_env)

    for binding in bindings:
        local_env['bindings'][binding['lexeme']] = {'c_name': binding['lexeme']}

    loop_code = _loop(envs, list(local_env['bindings'].keys()), exprs)

    f_params = ', '.join([f'Value {binding["lexeme"]}' for binding in bindings])
    if f_params:
        f_params = f'ObjMap* user_globals, {f_params}'
    else:
        f_params = 'ObjMap* user_globals'

    if not f_name:
        f_name = _get_generated_name(base='fn', envs=envs)
    envs[0]['functions'][f_name] = 'Value %s(%s) {\n  %s\n}' % (f_name, f_params, loop_code)

    envs.pop()

    return f_name


def defn_c(params, envs):
    name = params[0]['lexeme']
    f_name = _get_generated_name(base=f'u_{name}', envs=envs)

    current_ns = envs[0]['current_ns']
    envs[0]['namespaces'][current_ns][name] = {'type': 'function', 'c_name': f_name}

    fn_result = fn_c(params[1:], envs, f_name=f_name)

    return ''


def readline_c(params, envs):
    result_name = _get_generated_name('readline_result', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['code'].append(f'  Value {result_name} = readline();')
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))
    return result_name


def cli_args_c(params, envs):
    return 'OBJ_VAL(cli_args)'


def file_open_c(params, envs):
    path = compile_form(params[0], envs=envs)['code']
    mode = 'r'
    if len(params) > 1:
        if params[1]['lexeme'] == 'w':
            mode = 'w'
        else:
            raise Exception(f'invalid open mode: {params[1]}')
    result_name = _get_generated_name('file_obj', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['code'].append(f'  Value {result_name} = file_open({path}, "{mode}");')
    return result_name


def file_read_c(params, envs):
    file_obj = compile_form(params[0], envs=envs)['code']
    result_name = _get_generated_name('file_data', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['code'].append(f'  Value {result_name} = file_read({file_obj});')
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))
    return result_name


def file_write_c(params, envs):
    file_obj = compile_form(params[0], envs=envs)['code']
    data = compile_form(params[1], envs=envs)['code']
    result_name = _get_generated_name('file_write_result', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['code'].append(f'  Value {result_name} = file_write({file_obj}, {data});')
    return result_name


def file_close_c(params, envs):
    file_obj = compile_form(params[0], envs=envs)['code']
    result_name = _get_generated_name('file_close_result', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['code'].append(f'  Value {result_name} = file_close({file_obj});')
    return result_name


def os_mkdir_c(params, envs):
    path = compile_form(params[0], envs=envs)['code']
    result_name = _get_generated_name('dir_name', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['code'].append(f'  Value {result_name} = os_mkdir({path});')
    return result_name


def sqlite3_version_c(params, envs):
    result_name = _get_generated_name('sqlite3_version_s', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[0]['use_sqlite3'] = True
    envs[-1]['code'].append(f'  Value {result_name} = lang_sqlite3_version();')
    envs[-1]['post'].append(f'  if (IS_OBJ({result_name})) ' + '{\n' + f'    dec_ref_and_free(AS_OBJ({result_name}));\n' + '  }')
    return result_name


def sqlite3_open_c(params, envs):
    file_name = compile_form(params[0], envs=envs)['code']
    result_name = _get_generated_name('sqlite3_db', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[0]['use_sqlite3'] = True
    envs[-1]['code'].append(f'  Value {result_name} = lang_sqlite3_open({file_name});')
    return result_name


def sqlite3_close_c(params, envs):
    db = compile_form(params[0], envs=envs)['code']
    result_name = _get_generated_name('sqlite3_close_result', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[0]['use_sqlite3'] = True
    envs[-1]['code'].append(f'  Value {result_name} = lang_sqlite3_close({db});')
    return result_name


def sqlite3_execute_c(params, envs):
    db = compile_form(params[0], envs=envs)['code']
    sql_code = compile_form(params[1], envs=envs)['code']
    result_name = _get_generated_name('sqlite3_execute_result', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[0]['use_sqlite3'] = True
    envs[-1]['code'].append(f'  Value {result_name} = lang_sqlite3_execute({db}, {sql_code});')
    return result_name


global_ns = {
    'nil?': {'function': nil_c},
    '+': {'function': add_c},
    '-': {'function': subtract_c},
    '*': {'function': multiply_c},
    '/': {'function': divide_c},
    '=': {'function': equal_c},
    '>': {'function': greater_c, 'c_name': '*greater'},
    '>=': {'function': greater_equal_c},
    '<': {'function': less_c},
    '<=': {'function': less_equal_c},
    'to-number': {'function': to_number_c},
    'hash': {'function': hash_c},
    'print': {'function': print_c},
    'println': {'function': println_c},
    'count': {'function': count_c},
    'nth': {'function': nth_c},
    'conj': {'function': conj_c},
    'remove': {'function': remove_c},
    'sort': {'function': sort_c},
    'get': {'function': map_get_c},
    'contains?': {'function': map_contains_c},
    'assoc': {'function': map_assoc_c},
    'dissoc': {'function': map_dissoc_c},
    'keys': {'function': map_keys_c},
    'vals': {'function': map_vals_c},
    'pairs': {'function': map_pairs_c},
    'def': {'function': def_c},
    'let': {'function': let_c},
    'loop': {'function': loop_c},
    'fn': {'function': fn_c},
    'defn': {'function': defn_c},
    'read-line': {'function': readline_c},
    'cli-args': {'function': cli_args_c},
    'str': {'function': str_c},
    'file/open': {'function': file_open_c},
    'file/read': {'function': file_read_c},
    'file/write': {'function': file_write_c},
    'file/close': {'function': file_close_c},
}

language_math_ns = {
    'gcd': {'function': math_gcd_c},
}

language_string_ns = {
    'split': {'function': str_split_c},
    'lower': {'function': str_lower_c},
    'blank?': {'function': str_blank_c},
    'index-of': {'function': str_index_of_c},
    'subs': {'function': str_subs_c},
}

language_os_ns = {
    'mkdir': {'function': os_mkdir_c},
}

language_sqlite3_ns = {
    'version': {'function': sqlite3_version_c},
    'open': {'function': sqlite3_open_c},
    'close': {'function': sqlite3_close_c},
    'execute': {'function': sqlite3_execute_c},
}


character_replacements = {
    '-': '_M_',
    '?': '_Q_',
    '!': '_E_',
}


def _get_generated_name(base, envs):
    for c, replacement in character_replacements.items():
        base = base.replace(c, replacement)

    env = envs[0]
    if base not in env['functions'] and base not in env['temps'] and base not in env['namespaces']['user'] and base not in envs[-1].get('temps', set()):
        return base
    i = 1
    while True:
        name = f'{base}_{i}'
        if name not in env['functions'] and name not in env['temps'] and base not in env['namespaces']['user'] and name not in envs[-1].get('temps', set()):
            return name
        i += 1


def new_string_c(s, envs):
    name = _get_generated_name('str', envs=envs)

    envs[-1]['temps'].add(name)
    envs[-1]['code'].append(f'  Value {name} = copy_string("{s}", {len(s)});')
    envs[-1]['code'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    inc_ref(AS_OBJ({name}));' + '\n  }')
    envs[-1]['post'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'    dec_ref_and_free(AS_OBJ({name}));' + '\n  }')
    return name


def new_vector_c(v, envs):
    name = _get_generated_name('lst', envs=envs)
    envs[-1]['temps'].add(name)
    num_items = len(v)
    c_code = f'  Value {name} = OBJ_VAL(allocate_list((uint32_t) {num_items}));\n  inc_ref(AS_OBJ({name}));'
    c_items = [compile_form(item, envs=envs)['code'] for item in v]
    for c_item in c_items:
        c_code += f'\n  list_add(AS_LIST({name}), {c_item});'

    envs[-1]['code'].append(f'{c_code}\n')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def new_map_c(node, envs):
    name = _get_generated_name('map', envs=envs)
    envs[-1]['temps'].add(name)
    c_code = f'  Value {name} = OBJ_VAL(allocate_map());\n  inc_ref(AS_OBJ({name}));'
    keys = [compile_form(k, envs=envs)['code'] for k in node[::2]]
    values = [compile_form(v, envs=envs)['code'] for v in node[1::2]]
    c_items = zip(keys, values)

    for key, value in c_items:
        c_code += f'\n  map_set(AS_MAP({name}), {key}, {value});'

    envs[-1]['code'].append(f'{c_code}\n')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def _find_symbol(symbol, envs):
    current_ns = envs[0]['current_ns']
    symbol_name = symbol['lexeme']
    if symbol_name in global_ns:
        return global_ns[symbol_name]
    elif symbol_name in envs[0]['namespaces'][current_ns]:
        return envs[0]['namespaces'][current_ns][symbol_name]
    elif '/' in symbol_name:
        refer, name = symbol_name.split('/')
        if refer:
            for referred_as, ns in envs[0]['namespaces'].items():
                if refer == referred_as:
                    if name in ns:
                        return ns[name]
    for env in envs:
        if symbol_name in env.get('bindings', {}):
            if env['bindings'][symbol_name]:
                return env['bindings'][symbol_name]
            else:
                # show that we found it, but all we have to return is the symbol itself
                return symbol


def compile_form(node, envs):
    type_ = node['type']
    if type_ == 'nil':
        node['code'] = 'NIL_VAL'
        return node
    elif type_ == 'true':
        node['code'] = 'BOOL_VAL(true)'
        return node
    elif type_ == 'false':
        node['code'] = 'BOOL_VAL(false)'
        return node
    elif type_ == 'number':
        node['code'] = f'NUMBER_VAL({node["lexeme"]})'
        return node
    elif type_ == 'string':
        name = new_string_c(node['lexeme'], envs=envs)
        node['code'] = name
        return node
    elif type_ == 'ratio':
        numerator, denominator = node['lexeme'].split('/')
        node['code'] = f'ratio_val({numerator}, {denominator})'
        return node
    elif type_ == 'map':
        name = new_map_c(node['nodes'], envs=envs)
        node['code'] = name
        return node
    elif type_ == 'vector':
        name = new_vector_c(node['nodes'], envs=envs)
        node['code'] = name
        return node
    elif type_ == 'symbol':
        symbol = _find_symbol(node, envs)
        if symbol:
            if symbol.get('type') == 'var':
                symbol_name = node['lexeme']
                if '/' in symbol_name:
                    full_reference = symbol_name
                else:
                    full_reference = f'{envs[0]["current_ns"]}/{symbol_name}'
                name = _get_generated_name('user_global_lookup', envs=envs)
                envs[0]['temps'].add(name)
                code = f'  Value {name} = copy_string("{full_reference}", {len(full_reference)});'
                code += '\n  ' + f'if (IS_OBJ({name})) ' + '{\n' + f'    inc_ref(AS_OBJ({name}));' + '\n  }'
                envs[-1]['code'].append(code)
                envs[-1]['post'].append(f'  if (IS_OBJ({name})) ' + '{\n' + f'  dec_ref_and_free(AS_OBJ({name}));' + '\n  }')
                node['code'] = f'map_get(user_globals, {name}, NIL_VAL)'
                return node
            elif 'c_name' in symbol:
                node['code'] = symbol['c_name']
                return node
        raise Exception(f'unhandled symbol: {node}')
    elif type_ == 'list':
        first = node['nodes'][0]
        rest = node['nodes'][1:]
        if first['type'] == 'list':
            results = [compile_form(n, envs=envs) for n in node['nodes']]
            args = 'user_globals'
            if len(results) > 1:
                args += ', ' + ', '.join([r['code'] for r in results[1:]])
            result_name = _get_generated_name('fn_result', envs=envs)
            envs[-1]['code'].append(f'  Value {result_name} = {results[0]["code"]}({args});')
            envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))
            node['code'] = result_name
            return node
        elif first['type'] == 'symbol':
            if first['lexeme'] == 'recur':
                node['nodes'] = [first] + [compile_form(r, envs=envs) for r in rest]
                return node
            if first['lexeme'] == 'if':
                node['code'] = if_form_c(rest, envs=envs)
                return node

            symbol = _find_symbol(first, envs)
            if symbol:
                if 'function' in symbol and callable(symbol['function']):
                    node['code'] = symbol['function'](rest, envs=envs)
                    return node
                elif 'c_name' in symbol:
                    f_name = symbol['c_name']
                    results = [compile_form(n, envs=envs) for n in rest]
                    args = 'user_globals'
                    if results:
                        args += ', ' + ', '.join([r['code'] for r in results])
                    result_name = _get_generated_name('u_f_result', envs=envs)
                    envs[-1]['temps'].add(result_name)
                    envs[-1]['code'].append(f'  Value {result_name} = {f_name}({args});')
                    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))
                    node['code'] = result_name
                    return node
                else:
                    raise Exception(f'symbol first in list and not callable: {first} -- {symbol}')

            if first['lexeme'] == 'for':
                bindings = rest[0]['nodes']
                binding_name = bindings[0]['lexeme']
                c_name = _get_generated_name(f'u_{binding_name}', envs=envs)
                envs[-1]['bindings'][binding_name] = {'c_name': c_name}
                lst = compile_form(bindings[1], envs=envs)
                lst_name = _get_generated_name('tmp_lst', envs=envs)
                lst_count = _get_generated_name('tmp_lst_count', envs=envs)
                envs[-1]['temps'].add(lst_name)
                envs[-1]['temps'].add(lst_count)
                envs[-1]['code'].append(f'  ObjList* {lst_name} = AS_LIST({lst["code"]});')
                envs[-1]['code'].append('  for(uint32_t i=0; i<%s->count; i++) {\n' % lst_name)
                envs[-1]['code'].append(f'    Value {c_name} = {lst_name}->values[i];')
                local_env = {'temps': envs[-1]['temps'], 'code': [], 'post': [], 'bindings': {}}
                envs.append(local_env)
                for expr in rest[1:]:
                    compile_form(expr, envs=envs)
                code_lines = []
                if local_env['code']:
                    code_lines.extend(local_env['code'])
                if local_env['post']:
                    code_lines.extend(local_env['post'])
                envs.pop()
                for code_line in code_lines:
                    envs[-1]['code'].append(code_line)
                envs[-1]['code'].append('  }')
                node['code'] = 'NIL_VAL'
                return node
            elif first['lexeme'] == 'do':
                do_exprs = [compile_form(n, envs=envs) for n in rest]

                do_result = _get_generated_name('do_result', envs)

                f_code = f'  Value {do_result} = NIL_VAL;'
                if 'type' not in do_exprs[-1]:
                    raise Exception(f'no type: {do_exprs[-1]}')
                if do_exprs[-1]['type'] == 'list' and do_exprs[-1]['nodes'][0]['type'] == 'symbol' and do_exprs[-1]['nodes'][0]['lexeme'] == 'recur':
                    recur_name = envs[0]['recur_points'].pop()
                    for r in do_exprs[-1]['nodes'][1:]:
                        f_code += f'\n  recur_add(AS_RECUR({recur_name}), {r["code"]});'
                    f_code += f'\n  {do_result} = {recur_name};'
                else:
                    f_code += f'\n  {do_result} = {do_exprs[-1]["code"]};'
                    f_code += '\n  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }' % (do_result, do_result)

                envs[-1]['code'].append(f_code)
                envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n  dec_ref_and_free(AS_OBJ(%s));\n  }' % (do_result, do_result))
                node['code'] = do_result
                return node
            elif first['lexeme'] == 'with':
                bindings = rest[0]['nodes']
                paired_bindings = []
                for i in range(0, len(bindings), 2):
                    paired_bindings.append(bindings[i:i+2])
                for binding in paired_bindings:
                    result = compile_form(binding[1], envs=envs)
                    binding_name = _get_generated_name(base=binding[0]['lexeme'], envs=envs)
                    result['c_name'] = binding_name
                    envs[-1]['bindings'][binding[0]['lexeme']] = result
                    envs[-1]['code'].append(f'  Value {binding_name} = {result["code"]};\n')
                exprs = [compile_form(n, envs=envs) for n in rest[1:]]
                result = _get_generated_name('with_result', envs)
                f_code = f'  Value {result} = NIL_VAL;'
                if exprs[-1]['type'] == 'list' and exprs[-1]['nodes'][0]['type'] == 'symbol' and exprs[-1]['nodes'][0]['lexeme'] == 'recur':
                    recur_name = envs[0]['recur_points'].pop()
                    for r in exprs[-1]['nodes'][1:]:
                        f_code += f'\n  recur_add(AS_RECUR({recur_name}), {r["code"]});'
                    f_code += f'\n  {result} = {recur_name};'
                else:
                    f_code += f'\n  {result} = {exprs[-1]["code"]};'
                    f_code += '\n  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }' % (result, result)
                # add destructors
                for binding in paired_bindings:
                    if binding[1]['nodes'][0]['lexeme'] == 'sqlite3/open':
                        f_code += f'\n  lang_sqlite3_close({envs[-1]["bindings"][binding[0]["lexeme"]]["c_name"]});'
                    else:
                        raise Exception(f'unrecognized with constructor: {binding}')

                envs[-1]['code'].append(f_code)
                envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result, result))
                node['code'] = result
                return node
            elif first['lexeme'] == 'not':
                result = compile_form(rest[0], envs=envs)
                node['code'] = f'BOOL_VAL(!is_truthy({result["code"]}))'
                return node
            elif first['lexeme'] == 'and':
                params = [compile_form(r, envs=envs) for r in rest]
                num_params = len(params)
                and_result = _get_generated_name('and_result', envs)
                and_params = _get_generated_name('and_params', envs)
                envs[-1]['temps'].add(and_result)
                envs[-1]['temps'].add(and_params)
                envs[-1]['code'].append(f'  Value {and_params}[{num_params}];')
                for index, p in enumerate(params):
                    envs[-1]['code'].append(f'  {and_params}[{index}] = {p["code"]}; ')
                envs[-1]['code'].append(f'  Value {and_result} = BOOL_VAL(true);')
                envs[-1]['code'].append('  for (int i = 0; i<%s; i++) {' % num_params)
                envs[-1]['code'].append('    %s = %s[i];' % (and_result, and_params))
                envs[-1]['code'].append('    if(!is_truthy(%s)) { break; }' % and_result)
                envs[-1]['code'].append('  }')
                node['code'] = and_result
                return node
            elif first['lexeme'] == 'or':
                params = [compile_form(r, envs=envs) for r in rest]
                num_params = len(params)
                or_result = _get_generated_name('or_result', envs)
                or_params = _get_generated_name('or_params', envs)
                envs[-1]['temps'].add(or_result)
                envs[-1]['temps'].add(or_params)
                envs[-1]['code'].append(f'  Value {or_params}[{num_params}];')
                for index, p in enumerate(params):
                    envs[-1]['code'].append(f'  {or_params}[{index}] = {p["code"]}; ')
                envs[-1]['code'].append(f'  Value {or_result} = BOOL_VAL(true);')
                envs[-1]['code'].append('  for (int i = 0; i<%s; i++) {' % num_params)
                envs[-1]['code'].append('    %s = %s[i];' % (or_result, or_params))
                envs[-1]['code'].append('    if(is_truthy(%s)) { break; }' % or_result)
                envs[-1]['code'].append('  }')
                node['code'] = or_result
                return node
            elif first['lexeme'] == 'require':
                for require in rest:
                    if require['type'] == 'vector':
                        if require['nodes'][0]['type'] == 'symbol':
                            module_name = require['nodes'][0]['lexeme']
                        else:
                            if require['nodes'][0]['type'] == 'string':
                                module_name = require['nodes'][0]['lexeme']
                            else:
                                raise Exception(f'unhandled require type: {require["nodes"][0]}')
                        referred_as = require['nodes'][1]['lexeme']
                    else:
                        raise Exception(f'require argument needs to a vector')
                    if referred_as in envs[0]['namespaces']:
                        continue # already required, nothing to do

                    # find the module
                    if module_name.startswith('language.'):
                        if module_name == 'language.string':
                            envs[0]['namespaces'][referred_as] = copy.deepcopy(language_string_ns)
                        elif module_name == 'language.math':
                            envs[0]['namespaces'][referred_as] = copy.deepcopy(language_math_ns)
                        elif module_name == 'language.sqlite3':
                            envs[0]['namespaces'][referred_as] = copy.deepcopy(language_sqlite3_ns)
                        elif module_name == 'language.os':
                            envs[0]['namespaces'][referred_as] = copy.deepcopy(language_os_ns)
                        else:
                            raise Exception(f'system module {module_name} not found')
                    else:
                        # module_name is the file
                        if os.path.exists(module_name):
                            with open(module_name, 'rb') as module_file:
                                module_code = module_file.read().decode('utf8')
                            old_ns = envs[0]['current_ns']
                            envs[0]['current_ns'] = referred_as
                            envs[0]['namespaces'][referred_as] = {}
                            _compile_forms(module_code, program=envs[0], source_file=module_name)
                            envs[0]['current_ns'] = old_ns
                        else:
                            raise Exception(f'module {module_name} not found')
                node['code'] = ''
                return node
            else:
                raise Exception(f'unhandled symbol: {first}')
        else:
            raise Exception(f'unhandled list: {node}')
    else:
        raise Exception(f'unhandled node type: {type_}')


def _compile_forms(source, program=None, source_file=None):
    ast = parse(source)

    if not program:
        program = {
            'use_sqlite3': False,
            'namespaces': {'user': {}}, #including required & default user ns
            'current_ns': 'user',
            'functions': {},
            'init': [],
            'code': [],
            'post': [],
            'temps': set(),
            'recur_points': [],
            'bindings': {},
        }
    for f in ast['nodes']:
        compile_form(f, envs=[program])

    return program


def _compile(source, source_file=None):
    program = _compile_forms(source, source_file=source_file)

    c_code = INCLUDES

    if platform.system() == 'Windows':
        c_code += '#include <direct.h>\n'
        c_code += '#define WINDOWS 1\n'
    else:
        c_code += '#include <sys/stat.h>\n'

    if program['use_sqlite3']:
        c_code += f'#include "sqlite3.h"\n\n'
        c_code += '#define USE_SQLITE3 1'

    c_code += LANG_C_CODE

    if program['use_sqlite3']:
        c_code += f'{LANG_SQLITE_CODE}\n'

    c_code += '\n\n/* CUSTOM CODE */\n\n'

    if program['functions']:
        c_code += '\n\n'.join([f for f in program['functions'].values()]) + '\n\n'

    c_code += 'int main(int argc, char *argv[])\n{'
    c_code += '\n  cli_args = allocate_list((uint32_t) argc);'
    c_code += '\n  for (int i = 0; i < argc; i++) {\n    list_add(cli_args, copy_string(argv[i], (uint32_t) strlen(argv[i])));\n  }'
    c_code += '\n  ObjMap* user_globals = allocate_map();\n'

    if program['init']:
        c_code += '\n'.join(program['init'])

    for referred_as, ns in program['namespaces'].items():
        for name, value in ns.items():
            if value.get('type') == 'var':
                full_reference = f'{referred_as}/{name}'
                c_code += f'\n  map_set(user_globals, copy_string("{full_reference}", {len(full_reference)}), {value["code"]});\n'

    c_code += '\n' + '\n'.join(program['code'])

    if program['post']:
        c_code += '\n' + '\n'.join(program['post'])
    c_code += '\n  free_object((Obj*)user_globals);'
    c_code += '\n  free_object((Obj*)cli_args);'
    c_code += '\n  return 0;\n}'

    return c_code


GCC_CMD = 'gcc'
CLANG_CMD = 'clang'

# See https://github.com/airbus-seclab/c-compiler-security
GCC_CHECK_OPTIONS = [
    '-O2',
    '-std=c99',
    '-Werror',
    '-Wall',
    '-Wextra',
    '-Wno-error=unused-parameter',
    '-Wno-error=unused-variable',
    '-pedantic',
    '-Wpedantic',
    '-Wformat=2',
    '-Wformat-overflow=2',
    '-Wformat-truncation=2',
    '-Wformat-security',
    '-Wnull-dereference',
    '-Wstack-protector',
    '-Wtrampolines',
    '-Walloca',
    '-Wvla',
    '-Warray-bounds=2',
    '-Wimplicit-fallthrough=3',
    # '-Wtraditional-conversion',
    '-Wshift-overflow=2',
    '-Wcast-qual',
    '-Wstringop-overflow=4',
    '-Wconversion',
    '-Warith-conversion',
    '-Wlogical-op',
    '-Wduplicated-cond',
    '-Wduplicated-branches',
    '-Wformat-signedness',
    '-Wshadow',
    '-Wstrict-overflow=4',
    '-Wundef',
    '-Wstrict-prototypes',
    '-Wswitch-default',
    '-Wswitch-enum',
    '-Wstack-usage=1000000',
    # '-Wcast-align=strict',
    '-D_FORTIFY_SOURCE=2',
    '-fstack-protector-strong',
    '-fstack-clash-protection',
    '-fPIE',
    '-Wl,-z,relro',
    '-Wl,-z,now',
    '-Wl,-z,noexecstack',
    '-Wl,-z,separate-code',
    '-fsanitize=address',
    '-fsanitize=pointer-compare',
    '-fsanitize=pointer-subtract',
    '-fsanitize=leak',
    '-fno-omit-frame-pointer',
    '-fsanitize=undefined',
    '-fsanitize=bounds-strict',
    '-fsanitize=float-divide-by-zero',
    '-fsanitize=float-cast-overflow',
]
GCC_CHECK_ENV = {
    'ASAN_OPTIONS': 'strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:detect_invalid_pointer_pairs=2',
    'PATH': os.environ.get('PATH', ''),
}

CLANG_CHECK_OPTIONS = [
    '-O2',
    '-std=c99',
    '-Werror',
    '-Walloca',
    '-Wcast-qual',
    '-Wconversion',
    '-Wformat=2',
    '-Wformat-security',
    '-Wnull-dereference',
    '-Wstack-protector',
    '-Wvla',
    '-Warray-bounds',
    '-Warray-bounds-pointer-arithmetic',
    '-Wassign-enum',
    '-Wbad-function-cast',
    '-Wconditional-uninitialized',
    '-Wconversion',
    '-Wfloat-equal',
    '-Wformat-type-confusion',
    '-Widiomatic-parentheses',
    '-Wimplicit-fallthrough',
    '-Wloop-analysis',
    '-Wpointer-arith',
    '-Wshift-sign-overflow',
    '-Wshorten-64-to-32',
    '-Wswitch-enum',
    '-Wtautological-constant-in-range-compare',
    '-Wunreachable-code-aggressive',
    '-Wthread-safety',
    '-Wthread-safety-beta',
    '-Wcomma',
    '-D_FORTIFY_SOURCE=3',
    '-fstack-protector-strong',
    '-fPIE',
    '-fstack-clash-protection',
    '-fsanitize=bounds',
    '-fsanitize-undefined-trap-on-error',
    '-Wl,-z,relro',
    '-Wl,-z,now',
    '-Wl,-z,noexecstack',
    '-Wl,-z,separate-code',
    '-fsanitize=address',
    '-fsanitize=leak',
    '-fno-omit-frame-pointer',
    '-fsanitize=undefined',
    '-fsanitize=float-divide-by-zero',
    '-fsanitize=float-cast-overflow',
    '-fsanitize=integer',
]
CLANG_CHECK_ENV = {
    'ASAN_OPTIONS': 'strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:detect_invalid_pointer_pairs=2',
    'PATH': os.environ.get('PATH', ''),
}


def _get_compile_cmd_env(file_name, output_file_name, cc=None, with_checks=False):
    with open(file_name, 'rb') as f:
        source = f.read().decode('utf8')

    if '#define USE_SQLITE3 1' in source:
        c_sources = [os.path.join('lib', 'sqlite3.c')]
        includes = ['-I%s' % os.path.join('.', 'include')]
        libs = ['-Wl,-lm,-lpthread,-ldl']
    else:
        c_sources = []
        includes = []
        libs = ['-Wl,-lm']

    if cc:
        compiler = [cc]
    elif os.environ.get('CC'):
        compiler = [os.environ['CC']]
    else:
        compiler = [CLANG_CMD]

    env = None

    if with_checks:
        if 'clang' in compiler[0]:
            compiler.extend(CLANG_CHECK_OPTIONS)
            env = CLANG_CHECK_ENV
        elif 'gcc' in compiler[0]:
            compiler.extend(GCC_CHECK_OPTIONS)
            env = GCC_CHECK_ENV
        else:
            raise RuntimeError(f'no checks to use for building with {compiler[0]}')
    else:
        compiler.extend(['-O2', '-std=c99'])

    cmd = compiler + [*includes, '-o', output_file_name, *c_sources, file_name, *libs]

    return cmd, env


def build_executable(file_name, output_file_name, with_checks=False):
    if os.path.exists(output_file_name):
        print(f'{output_file_name} already exists')
        sys.exit(1)

    compile_cmd, env = _get_compile_cmd_env(file_name, output_file_name, with_checks)
    try:
        subprocess.run(compile_cmd, check=True, env=env, capture_output=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                data = f.read().decode('utf8')
                print(data)
        print(e)
        print(e.stderr.decode('utf8'))
        sys.exit(1)


def run_executable(file_name, cli_args=None):
    if not file_name.startswith('/'):
        file_name = f'./{file_name}'
    cmd = [file_name]
    if cli_args:
        cmd.extend(cli_args)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)


def compile_to_c(file_name, output_file_name):
    with open(file_name, 'rb') as f:
        source = f.read().decode('utf8')

    c_program = _compile(source, source_file=file_name)

    with open(output_file_name, mode='wb') as f:
        f.write(c_program.encode('utf8'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Random language')
    parser.add_argument('-c', action='store_true', dest='compile', help='compile the file to C')
    parser.add_argument('-b', action='store_true', dest='build', help='build executable')
    parser.add_argument('-r', action='store_true', dest='run', help='compile, build, & run')
    parser.add_argument('-o', action='store', dest='output', help='output file')
    parser.add_argument('--with-checks', action='store_true', dest='with_checks', help='build with compiler checks')
    parser.add_argument('file', type=str, nargs='?', help='file to interpret')
    parser.add_argument('cli_args', type=str, nargs='*', default=[], help='args for running the compiled/built executable')

    args = parser.parse_args()

    if args.compile:
        if args.file:
            if args.output:
                c_file_name = args.output
            else:
                tmp = tempfile.mkdtemp(dir='.')
                c_file_name = os.path.join(tmp, 'code.c')
            compile_to_c(Path(args.file), c_file_name)
            print(f'Compiled to {c_file_name}')
        else:
            print('no file to compile')
    elif args.build:
        if args.file:
            if args.output:
                executable = args.output
            else:
                tmp = tempfile.mkdtemp(dir='.')
                executable = os.path.join(tmp, 'program')
            if not args.file.endswith('.c'):
                with tempfile.TemporaryDirectory() as c_tmp:
                    c_file_name = os.path.join(c_tmp, 'code.c')
                    compile_to_c(Path(args.file), c_file_name)
                    build_executable(c_file_name, output_file_name=executable, with_checks=args.with_checks)
            else:
                c_file_name = args.file
                build_executable(c_file_name, output_file_name=executable, with_checks=args.with_checks)
            print(f'Built executable at {executable}')
        else:
            print('no file to build')
    elif args.run:
        if args.file:
            with tempfile.TemporaryDirectory(dir='.') as tmp:
                executable = os.path.join(tmp, 'program')
                if args.file.endswith('.c'):
                    build_executable(args.file, output_file_name=executable, with_checks=args.with_checks)
                    run_executable(executable, cli_args=args.cli_args)
                else:
                    c_file_name = os.path.join(tmp, 'code.c')
                    compile_to_c(Path(args.file), c_file_name)
                    build_executable(c_file_name, output_file_name=executable, with_checks=args.with_checks)
                    run_executable(executable, cli_args=args.cli_args)
        else:
            print('no file to run')
