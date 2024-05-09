
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>


#define ALLOCATE(type, count)     (type*)reallocate(NULL, sizeof(type) * (count))

#define ALLOCATE_OBJ(type, objectType)     (type*)allocateObject(sizeof(type), objectType)

#define GROW_CAPACITY(capacity)             ((capacity) < 8 ? 8 : (capacity) * 2)

#define GROW_ARRAY(type, pointer, oldCount, newCount)             (type*)reallocate(pointer, sizeof(type) * (newCount))

#define FREE(type, pointer) reallocate(pointer, (size_t)0)

#define FREE_ARRAY(type, pointer)             reallocate(pointer, (size_t)0)

#define NIL_VAL  ((Value){NIL, {.boolean = 0}})
#define TOMBSTONE_VAL  ((Value){TOMBSTONE, {.boolean = 0}})
#define BOOL_VAL(value)  ((Value){BOOL, {.boolean = value}})
#define NUMBER_VAL(value)  ((Value){NUMBER, {.number = value}})
#define RECUR_VAL(value)  ((Value){RECUR, {.recur = value}})
#define FILE_VAL(value)   ((Value){FILE_HANDLE, {.file = (FILE*)value}})
#define OBJ_VAL(object)   ((Value){OBJ, {.obj = (Obj*)object}})
#define AS_BOOL(value)  ((value).data.boolean)
#define AS_NUMBER(value)  ((value).data.number)
#define AS_RATIO(value)  ((value).data.ratio)
#define AS_RECUR(value)       ((value).data.recur)
#define AS_FILE(value)       ((value).data.file)
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
#define ERROR_GENERAL '\x01'
#define ERROR_TYPE '\x02'
#define ERROR_DIVIDE_BY_ZERO '\x03'

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
  OBJ,
#if defined(USE_SQLITE3)
  SQLITE3_DB,
#endif
} ValueType;

typedef struct {
  unsigned char type;
  unsigned char message[7];
} ErrorInfo;

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
#if defined(USE_SQLITE3)
    sqlite3* db;
#endif
  } data;
} Value;

Value error_val(unsigned char type, char* message) {
  ErrorInfo info;
  info.type = type;
  if (strlen(message) > 5) {
    info.message[0] = (unsigned char) message[0];
    info.message[1] = (unsigned char) message[1];
    info.message[2] = (unsigned char) message[2];
    info.message[3] = (unsigned char) message[3];
    info.message[4] = (unsigned char) message[4];
    info.message[5] = (unsigned char) message[5];
  } else {
    info.message[0] = ' ';
    info.message[1] = ' ';
    info.message[2] = ' ';
    info.message[3] = ' ';
    info.message[4] = ' ';
    info.message[5] = ' ';
  }
  info.message[6] = '\0';
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

ObjMap* interned_strings;
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

static ObjString* allocate_string(char* chars, uint32_t length, uint32_t hash) {
  ObjString* string = ALLOCATE_OBJ(ObjString, OBJ_STRING);
  string->length = length;
  string->hash = hash;
  string->chars = chars;
  if (length < 4) {
    map_set(interned_strings, OBJ_VAL(string), NIL_VAL);
  }
  return string;
}

ObjString* find_interned_string(const char* chars, uint32_t length, uint32_t hash) {
  if (interned_strings->num_entries == 0) { return NULL; }
  uint32_t index = hash % (uint32_t)interned_strings->indices_capacity;
  for (;;) {
    if (interned_strings->indices[index] == MAP_EMPTY) {
      return NULL;
    }
    MapEntry entry = interned_strings->entries[interned_strings->indices[index]];
    ObjString* key_string = AS_STRING(entry.key);
    if (key_string->length == length &&
        key_string->hash == hash &&
        memcmp(key_string->chars, chars, (size_t)length) == 0) {
      // We found it.
      return key_string;
    }

    index = (index + 1) % (uint32_t)interned_strings->indices_capacity;
  }

  return NULL;
}

ObjString* copy_string(const char* chars, uint32_t length) {
  uint32_t hash = hash_string(chars, length);
  if (length < 4) {
    ObjString* interned = find_interned_string(chars, length, hash);
    if (interned != NULL) {
      return interned;
    }
  }
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

Value list_count(Value list) {
  return NUMBER_VAL((double) AS_LIST(list)->count);
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
  return error_val(ERROR_TYPE, "      ");
}

Value add_list(Value numbers) {
  ObjList* numbers_list = AS_LIST(numbers);
  Value item = numbers_list->values[0];
  if (IS_NUMBER(item)) {
    double result = AS_NUMBER(item);
    for (uint32_t i = 1; i < numbers_list->count; i++) {
      item = numbers_list->values[i];
      if (!IS_NUMBER(item)) {
        return error_val(ERROR_TYPE, "      ");
      }
      result += AS_NUMBER(item);
    }
    return NUMBER_VAL(result);
  } else if (IS_RATIO(item)) {
    Ratio result = AS_RATIO(item);
    for (uint32_t i = 1; i < numbers_list->count; i++) {
      item = numbers_list->values[i];
      if (!IS_RATIO(item)) {
        return error_val(ERROR_TYPE, "      ");
      }
      result = AS_RATIO(add_two_ratios(result, AS_RATIO(item)));
    }
    return ratio_val(result.numerator, result.denominator);
  }
  return error_val(ERROR_TYPE, "      ");
}

Value subtract_two(Value x, Value y) {
  if (!IS_NUMBER(x) || !IS_NUMBER(y)) {
    return error_val(ERROR_TYPE, "      ");
  }
  return NUMBER_VAL(AS_NUMBER(x) - AS_NUMBER(y));
}

Value subtract_list(Value numbers) {
  ObjList* numbers_list = AS_LIST(numbers);
  Value item = numbers_list->values[0];
  if (!IS_NUMBER(item)) {
    return error_val(ERROR_TYPE, "      ");
  }
  double result = AS_NUMBER(item);
  for (uint32_t i = 1; i < numbers_list->count; i++) {
    item = numbers_list->values[i];
    if (!IS_NUMBER(item)) {
      return error_val(ERROR_TYPE, "      ");
    }
    result = result - AS_NUMBER(item);
  }
  return NUMBER_VAL(result);
}

Value multiply_two(Value x, Value y) {
  if (!IS_NUMBER(x) || !IS_NUMBER(y)) {
    return error_val(ERROR_TYPE, "      ");
  }
  return NUMBER_VAL(AS_NUMBER(x) * AS_NUMBER(y));
}

Value multiply_list(Value numbers) {
  ObjList* numbers_list = AS_LIST(numbers);
  Value item = numbers_list->values[0];
  if (!IS_NUMBER(item)) {
    return error_val(ERROR_TYPE, "      ");
  }
  double result = AS_NUMBER(item);
  for (uint32_t i = 1; i < numbers_list->count; i++) {
    item = numbers_list->values[i];
    if (!IS_NUMBER(item)) {
      return error_val(ERROR_TYPE, "      ");
    }
    result = result * AS_NUMBER(item);
  }
  return NUMBER_VAL(result);
}

Value divide_two(Value x, Value y) {
  if (!IS_NUMBER(x) || !IS_NUMBER(y)) {
    return error_val(ERROR_TYPE, "      ");
  }
  if (double_equal(AS_NUMBER(y), 0)) {
    return error_val(ERROR_DIVIDE_BY_ZERO, "      ");
  }
  return NUMBER_VAL(AS_NUMBER(x) / AS_NUMBER(y));
}

Value divide_list(Value numbers) {
  ObjList* numbers_list = AS_LIST(numbers);
  Value item = numbers_list->values[0];
  if (!IS_NUMBER(item)) {
    return error_val(ERROR_TYPE, "      ");
  }
  double result = AS_NUMBER(item);
  for (uint32_t i = 1; i < numbers_list->count; i++) {
    item = numbers_list->values[i];
    if (!IS_NUMBER(item)) {
      return error_val(ERROR_TYPE, "      ");
    }
    if (double_equal(AS_NUMBER(item), 0)) {
      return error_val(ERROR_DIVIDE_BY_ZERO, "      ");
    }
    result = result / AS_NUMBER(item);
  }
  return NUMBER_VAL(result);
}

Value greater(ObjMap* user_globals, Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) > AS_NUMBER(y)); }
Value greater_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) >= AS_NUMBER(y)); }
Value less_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) <= AS_NUMBER(y)); }
Value less(ObjMap* user_globals, Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) < AS_NUMBER(y)); }

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
  if (!IS_STRING(key)) {
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
      printf("ERROR: DivideByZero - %s", value.data.err_info.message);
    } else if (value.data.err_info.type == ERROR_TYPE) {
      printf("ERROR: Type - %s", value.data.err_info.message);
    } else {
      printf("ERROR: General - %s", value.data.err_info.message);
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
  else {
    printf("%s", AS_CSTRING(value));
  }
  return NIL_VAL;
}

Value println(Value value) {
  print(value);
  printf("\n");
  return NIL_VAL;
}

ObjList* cli_args;

Value readline(void) {
  /* K&R p29 */
  int ch = 0;
  char buffer[MAX_LINE];
  uint32_t num_chars;
  for (num_chars=0; num_chars<(MAX_LINE-1) && (ch=getchar()) != EOF && ch != '\n'; num_chars++) {
    buffer[num_chars] = (char) ch;
  }
  if ((ch == EOF) && (num_chars == 0)) {
    return NIL_VAL;
  }
  Value result = OBJ_VAL(copy_string(buffer, num_chars));
  inc_ref(AS_OBJ(result));
  return result;
}

Value str_blank(Value string) {
  if (IS_NIL(string)) {
    return BOOL_VAL(true);
  }
  ObjString* s = AS_STRING(string);
  if (s->length == 0) {
    return BOOL_VAL(true);
  }
  for (int i = 0; s->chars[i] != '\0'; i++) {
    if (!isspace(s->chars[i])) {
      return BOOL_VAL(false);
    }
  }
  return BOOL_VAL(true);
}

Value str_lower(Value string) {
  ObjString* s = AS_STRING(string);
  ObjString* s_lower = copy_string(s->chars, s->length);
  for (int i=0; s_lower->chars[i] != '\0'; i++) {
    s_lower->chars[i] = (char) tolower((int) s_lower->chars[i]);
  }
  return OBJ_VAL(s_lower);
}

Value str_split(Value string) {
  ObjString* s = AS_STRING(string);
  ObjList* splits = allocate_list((uint32_t) 0);
  uint32_t split_length = 0;
  int split_start_index = 0;
  for (int i=0; s->chars[i] != '\0'; i++) {
    if (s->chars[i] == ' ') {
      ObjString* split = copy_string(&(s->chars[split_start_index]), split_length);
      list_add(splits, OBJ_VAL(split));
      split_start_index = i + 1;
      split_length = 0;
    }
    else {
      split_length++;
    }
  }
  ObjString* split = copy_string(&(s->chars[split_start_index]), split_length);
  list_add(splits, OBJ_VAL(split));
  return OBJ_VAL(splits);
}

Value str_str(Value v) {
  // if we got a string, no need to do any work, just return it
  if (IS_STRING(v)) {
    inc_ref(AS_OBJ(v));
    return v;
  }

  Value s;
  if (IS_BOOL(v)) {
    if (AS_BOOL(v)) {
      s = OBJ_VAL(copy_string("true", 4));
    }
    else {
      s = OBJ_VAL(copy_string("false", 5));
    }
  }
  else if (IS_NUMBER(v)) {
    char str[100];
    int32_t num_chars = sprintf(str, "%g", AS_NUMBER(v));
    s = OBJ_VAL(copy_string(str, (uint32_t) num_chars));
  }
  else if (IS_LIST(v)) {
    s = OBJ_VAL(copy_string("[]", 2));
  }
  else if (IS_MAP(v)) {
    s = OBJ_VAL(copy_string("{}", 2));
  }
  else {
    s = OBJ_VAL(copy_string("", 0));
  }
  inc_ref(AS_OBJ(s));
  return s;
}

Value str_join(Value list_val) {
  ObjList* list = AS_LIST(list_val);

  uint32_t num_bytes = 0;

  for (uint32_t i = 0; i < list->count; i++) {
    Value v = list_get(list_val, (int32_t)i);
    Value v_str = str_str(v);
    num_bytes = num_bytes + AS_STRING(v_str)->length;
    dec_ref_and_free(AS_OBJ(v_str));
    if (IS_OBJ(v)) {
      dec_ref_and_free(AS_OBJ(v));
    }
  }

  char* heapChars = ALLOCATE(char, (size_t)(num_bytes+1));
  char* start_char = heapChars;

  for (uint32_t i = 0; i < list->count; i++) {
    Value v = list_get(list_val, (int32_t)i);
    ObjString* s = AS_STRING(str_str(v));
    memcpy(start_char, s->chars, (size_t)s->length);
    start_char = start_char + s->length;
  }
  heapChars[num_bytes] = 0;
  uint32_t hash = hash_string(heapChars, num_bytes);
  return OBJ_VAL(allocate_string(heapChars, num_bytes, hash));
}

Value math_gcd(Value param_1, Value param_2) {
  if (!IS_NUMBER(param_1) || !IS_NUMBER(param_2)) {
    return error_val(ERROR_TYPE, "      ");
  }
  double p1 = AS_NUMBER(param_1);
  double p2 = AS_NUMBER(param_2);
  if (!is_integer(p1) || !is_integer(p2)) {
    return error_val(ERROR_TYPE, "      ");
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
  Value result = OBJ_VAL(copy_string(buffer, num_chars));
  inc_ref(AS_OBJ(result));
  return result;
}

Value file_write(Value file, Value data) {
  FILE* fp = AS_FILE(file);
  fprintf(fp, "%s", AS_CSTRING(data));
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


/* CUSTOM CODE */

Value let(ObjMap* user_globals, Value words, Value counts, Value u_word) {
    Value map_get_ = map_get(AS_MAP(counts), u_word, NUMBER_VAL(0));
  if (IS_OBJ(map_get_)) {
    inc_ref(AS_OBJ(map_get_));
  }
  Value cur_M_count = map_get_;

  Value add_result = add_two(cur_M_count, NUMBER_VAL(1));
  Value map_assoc = map_set(AS_MAP(counts), u_word, add_result);

  if (IS_OBJ(map_assoc)) {
    inc_ref(AS_OBJ(map_assoc));
  }
  if (IS_OBJ(map_get_)) {
    dec_ref_and_free(AS_OBJ(map_get_));
  }
  return map_assoc;
}

Value u_process_M_words(ObjMap* user_globals, Value counts, Value words) {
  
  ObjList* tmp_lst = AS_LIST(words);
    for(uint32_t i=0; i<tmp_lst->count; i++) {

      Value u_word = tmp_lst->values[i];
    Value if_result = NIL_VAL;
  Value str_blank_ = str_blank(u_word);
  if (is_truthy(BOOL_VAL(!is_truthy(str_blank_)))) {
    Value let_result = let(user_globals, words, counts, u_word);
      if_result = let_result;
    if (IS_OBJ(if_result)) {
      inc_ref(AS_OBJ(if_result));
    }
    if (IS_OBJ(let_result)) {
    dec_ref_and_free(AS_OBJ(let_result));
  }
  } // end true code

    if (IS_OBJ(if_result)) {
    dec_ref_and_free(AS_OBJ(if_result));
  }
    }
    Value result = NIL_VAL;
    if (IS_OBJ(result)) {
      inc_ref(AS_OBJ(result));
    }
      return result;
  return NIL_VAL;
}

Value let_1(ObjMap* user_globals, Value counts, Value line) {
    Value str_lower_ = str_lower(line);
  inc_ref(AS_OBJ(str_lower_));
  Value str_split_ = str_split(str_lower_);
  inc_ref(AS_OBJ(str_split_));
  Value words = str_split_;

  Value u_f_result = u_process_M_words(user_globals, counts, words);

  if (IS_OBJ(u_f_result)) {
    inc_ref(AS_OBJ(u_f_result));
  }
  dec_ref_and_free(AS_OBJ(str_lower_));
  dec_ref_and_free(AS_OBJ(str_split_));
  if (IS_OBJ(u_f_result)) {
    dec_ref_and_free(AS_OBJ(u_f_result));
  }
  return u_f_result;
}

Value u_process_M_line(ObjMap* user_globals, Value counts, Value line) {
  
  Value let_result_1 = let_1(user_globals, counts, line);
    Value result_1 = let_result_1;
    if (IS_OBJ(result_1)) {
      inc_ref(AS_OBJ(result_1));
    }
  if (IS_OBJ(let_result_1)) {
    dec_ref_and_free(AS_OBJ(let_result_1));
  }
      return result_1;
  return NIL_VAL;
}

Value u_compare(ObjMap* user_globals, Value a, Value b) {
  
    Value result_2 = greater(user_globals, list_get(a, (int32_t) AS_NUMBER(NUMBER_VAL(1))), list_get(b, (int32_t) AS_NUMBER(NUMBER_VAL(1))));
    if (IS_OBJ(result_2)) {
      inc_ref(AS_OBJ(result_2));
    }
      return result_2;
  return NIL_VAL;
}

Value loop(ObjMap* user_globals, Value counts) {

  Value readline_result = readline();
  Value line = readline_result;
  if (IS_OBJ(line)) {
    inc_ref(AS_OBJ(line));
  }
  if (IS_OBJ(readline_result)) {
    dec_ref_and_free(AS_OBJ(readline_result));
  }
  Recur recur_1;
  recur_init(&recur_1);
  Value recur = RECUR_VAL(&recur_1);
  bool continueFlag = false;
  do {
  Value if_result_1 = NIL_VAL;
  if (is_truthy(BOOL_VAL(!is_truthy(nil_Q_(line))))) {
    Value if_result_2 = NIL_VAL;
  Value str_blank_ = str_blank(line);
  if (is_truthy(BOOL_VAL(!is_truthy(str_blank_)))) {
    Value u_f_result_1 = u_process_M_line(user_globals, counts, line);
      if_result_2 = u_f_result_1;
    if (IS_OBJ(if_result_2)) {
      inc_ref(AS_OBJ(if_result_2));
    }
    if (IS_OBJ(u_f_result_1)) {
    dec_ref_and_free(AS_OBJ(u_f_result_1));
  }
  } // end true code

    Value readline_result_1 = readline();
    Value do_result = NIL_VAL;
  recur_add(AS_RECUR(recur), readline_result_1);
  do_result = recur;
      if_result_1 = do_result;
    if (IS_OBJ(if_result_1)) {
      inc_ref(AS_OBJ(if_result_1));
    }
    if (IS_OBJ(if_result_2)) {
    dec_ref_and_free(AS_OBJ(if_result_2));
  }
    if (IS_OBJ(readline_result_1)) {
    dec_ref_and_free(AS_OBJ(readline_result_1));
  }
    if (IS_OBJ(do_result)) {
  dec_ref_and_free(AS_OBJ(do_result));
  }
  } // end true code

    Value result_3 = if_result_1;
    if (IS_OBJ(result_3)) {
      inc_ref(AS_OBJ(result_3));
    }
  if (IS_OBJ(if_result_1)) {
    dec_ref_and_free(AS_OBJ(if_result_1));
  }
    if (IS_RECUR(result_3)) {
      /* grab values from result and update  */
      if (IS_OBJ(line)) {
      dec_ref_and_free(AS_OBJ(line));
    }
      line = recur_get(result_3, 0);
      if (IS_OBJ(line)) {
      inc_ref(AS_OBJ(line));
    }
    continueFlag = true;
    recur_free(&recur_1);
  }
    else {

      recur_free(&recur_1);
      return result_3;
    }
  } while (continueFlag);
  return NIL_VAL;
}

Value let_2(ObjMap* user_globals, Value counts) {
    Value map_pairs_ = map_pairs(AS_MAP(counts));
  inc_ref(AS_OBJ(map_pairs_));
  Value sortedlist = list_sort(user_globals, map_pairs_, u_compare);

  Value str = OBJ_VAL(copy_string(" ", 1));
  inc_ref(AS_OBJ(str));
  Value space = str;

  ObjList* tmp_lst_1 = AS_LIST(sortedlist);
  for(uint32_t i=0; i<tmp_lst_1->count; i++) {

    Value u_entry = tmp_lst_1->values[i];
  Value print_result = print(list_get(u_entry, (int32_t) AS_NUMBER(NUMBER_VAL(0))));
  Value print_result_1 = print(space);
  Value println_result = println(list_get(u_entry, (int32_t) AS_NUMBER(NUMBER_VAL(1))));
  }
  dec_ref_and_free(AS_OBJ(map_pairs_));
  dec_ref_and_free(AS_OBJ(str));
  return NIL_VAL;
}

Value let_3(ObjMap* user_globals) {
    Value map = OBJ_VAL(allocate_map());
  inc_ref(AS_OBJ(map));

  Value counts = map;

  Value loop_result = loop(user_globals, counts);
  Value let_result_2 = let_2(user_globals, counts);

  if (IS_OBJ(let_result_2)) {
    inc_ref(AS_OBJ(let_result_2));
  }
  dec_ref_and_free(AS_OBJ(map));
  if (IS_OBJ(loop_result)) {
    dec_ref_and_free(AS_OBJ(loop_result));
  }
  if (IS_OBJ(let_result_2)) {
    dec_ref_and_free(AS_OBJ(let_result_2));
  }
  return let_result_2;
}

int main(int argc, char *argv[])
{
  cli_args = allocate_list((uint32_t) argc);
  for (int i = 0; i < argc; i++) {
    list_add(cli_args, OBJ_VAL(copy_string(argv[i], (uint32_t) strlen(argv[i]))));
  }
  interned_strings = allocate_map();
  ObjMap* user_globals = allocate_map();

  Value let_result_3 = let_3(user_globals);
  if (IS_OBJ(let_result_3)) {
    dec_ref_and_free(AS_OBJ(let_result_3));
  }
  free_object((Obj*)user_globals);
  free_object((Obj*)interned_strings);
  free_object((Obj*)cli_args);
  return 0;
}