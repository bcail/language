#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>


#define ALLOCATE(type, count)     (type*)reallocate(NULL, sizeof(type) * (count))

#define ALLOCATE_OBJ(type, objectType)     (type*)allocateObject(sizeof(type), objectType)

#define GROW_CAPACITY(capacity)             ((capacity) < 8 ? 8 : (capacity) * 2)

#define GROW_ARRAY(type, pointer, oldCount, newCount)             (type*)reallocate(pointer, sizeof(type) * (newCount))

#define FREE(type, pointer) reallocate(pointer, (size_t)0)

#define FREE_ARRAY(type, pointer)             reallocate(pointer, (size_t)0)

#define NIL_VAL  ((Value){NIL, {.number = 0}})
#define BOOL_VAL(value)  ((Value){BOOL, {.boolean = value}})
#define NUMBER_VAL(value)  ((Value){NUMBER, {.number = value}})
#define RECUR_VAL(value)  ((Value){RECUR, {.recur = value}})
#define OBJ_VAL(object)   ((Value){OBJ, {.obj = (Obj*)object}})
#define AS_BOOL(value)  ((value).data.boolean)
#define AS_NUMBER(value)  ((value).data.number)
#define AS_OBJ(value)  ((value).data.obj)
#define AS_RECUR(value)       ((value).data.recur)
#define AS_STRING(value)       ((ObjString*)AS_OBJ(value))
#define AS_CSTRING(value)      (((ObjString*)AS_OBJ(value))->chars)
#define AS_LIST(value)       ((ObjList*)AS_OBJ(value))
#define AS_MAP(value)       ((ObjMap*)AS_OBJ(value))
#define IS_NIL(value)  ((value).type == NIL)
#define IS_BOOL(value)  ((value).type == BOOL)
#define IS_NUMBER(value)  ((value).type == NUMBER)
#define IS_RECUR(value)  ((value).type == RECUR)
#define IS_OBJ(value)  ((value).type == OBJ)
#define IS_STRING(value)  isObjType(value, OBJ_STRING)
#define IS_LIST(value)  isObjType(value, OBJ_LIST)
#define IS_MAP(value)  isObjType(value, OBJ_MAP)
#define MAP_EMPTY (-1)
#define MAP_MAX_LOAD 0.75
#define MAX_LINE 1000

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

typedef struct Obj Obj;

struct Obj {
  ObjType type;
  uint32_t ref_cnt;
  // struct Obj* next;
};

typedef enum {
  NIL,
  BOOL,
  NUMBER,
  RECUR,
  OBJ,
} ValueType;

typedef struct Recur Recur;

typedef struct {
  ValueType type;
  union {
    bool boolean;
    double number;
    Obj* obj;
    Recur* recur;
  } data;
} Value;

static inline bool isObjType(Value value, ObjType type) {
  return IS_OBJ(value) && AS_OBJ(value)->type == type;
}

typedef struct {
  Obj obj;
  size_t length;
  uint32_t hash;
  char* chars;
} ObjString;

struct Recur {
  size_t count;
  size_t capacity;
  Value* values;
};

typedef struct {
  Obj obj;
  size_t count;
  size_t capacity;
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
  uint32_t hash;
  Value key;
  Value value;
} MapEntry;

typedef struct {
  Obj obj;
  size_t num_entries;
  size_t indices_capacity;
  size_t entries_capacity;
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

static uint32_t hashString(const char* key, size_t length) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < length; i++) {
    hash ^= (uint8_t)key[i];
    hash *= 16777619;
  }
  return hash;
}

static ObjString* allocateString(char* chars, size_t length) {
  ObjString* string = ALLOCATE_OBJ(ObjString, OBJ_STRING);
  string->length = length;
  string->hash = hashString(chars, length);
  string->chars = chars;
  return string;
}

ObjString* copyString(const char* chars, size_t length) {
  char* heapChars = ALLOCATE(char, length + 1);
  memcpy(heapChars, chars, length);
  heapChars[length] = 0; /* terminate it w/ NULL, so we can pass c-string to functions that need it */
  return allocateString(heapChars, length);
}

ObjList* allocate_list(void) {
  ObjList* list = ALLOCATE_OBJ(ObjList, OBJ_LIST);
  list->count = 0;
  list->capacity = 0;
  list->values = NULL;
  return list;
}

void list_add(Value list_value, Value item) {
  ObjList* list = AS_LIST(list_value);
  if (list->capacity < list->count + 1) {
    size_t oldCapacity = list->capacity;
    list->capacity = GROW_CAPACITY(oldCapacity);
    list->values = GROW_ARRAY(Value, list->values, oldCapacity, list->capacity);
  }

  list->values[list->count] = item;
  list->count++;
  if (IS_OBJ(item)) {
    inc_ref(AS_OBJ(item));
  }
}

Value list_get(Value list, Value index) {
  if (AS_NUMBER(index) < 0) {
    return NIL_VAL;
  }
  /* size_t is the unsigned integer type returned by the sizeof operator */
  size_t num_index = (size_t) AS_NUMBER(index);
  if (num_index < AS_LIST(list)->count) {
    return AS_LIST(list)->values[num_index];
  }
  else {
    return NIL_VAL;
  }
}

Value list_count(Value list) {
  return NUMBER_VAL((int) AS_LIST(list)->count);
}

void swap(Value v[], size_t i, size_t j) {
  if (i == j) {
    return;
  }
  Value temp = v[i];
  v[i] = v[j];
  v[j] = temp;
}

Value add(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) + AS_NUMBER(y)); }
Value subtract(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) - AS_NUMBER(y)); }
Value multiply(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) * AS_NUMBER(y)); }
Value divide(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) / AS_NUMBER(y)); }
Value greater(ObjMap* user_globals, Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) > AS_NUMBER(y)); }
Value greater_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) >= AS_NUMBER(y)); }
Value less_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) <= AS_NUMBER(y)); }
Value less(ObjMap* user_globals, Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) < AS_NUMBER(y)); }

void quick_sort(ObjMap* user_globals, Value v[], size_t left, size_t right, Value (*compare) (ObjMap*, Value, Value)) {
  /* C Programming Language K&R p87*/
  size_t i, last;
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
  quick_sort(user_globals, lst->values, (size_t)0, (lst->count)-1, *compare);
  return OBJ_VAL(lst);
}

void recur_init(Recur* recur) {
  recur->count = 0;
  recur->capacity = 0;
  recur->values = NULL;
}

void recur_free(Recur* recur) {
  for (size_t i = 0; i < recur->count; i++) {
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
    size_t oldCapacity = recur->capacity;
    recur->capacity = GROW_CAPACITY(oldCapacity);
    recur->values = GROW_ARRAY(Value, recur->values, oldCapacity, recur->capacity);
  }

  recur->values[recur->count] = item;
  recur->count++;
  if (IS_OBJ(item)) {
    inc_ref(AS_OBJ(item));
  }
}

Value recur_get(Value recur, Value index) {
  /* size_t is the unsigned integer type returned by the sizeof operator */
  size_t num_index = (size_t) AS_NUMBER(index);
  if (num_index < AS_RECUR(recur)->count) {
    return AS_RECUR(recur)->values[num_index];
  }
  else {
    return NIL_VAL;
  }
}

Value recur_count(Value recur) {
  return NUMBER_VAL((int) AS_RECUR(recur)->count);
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
  return NUMBER_VAL((int) AS_MAP(map)->num_entries);
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
    double x_double = AS_NUMBER(x);
    double y_double = AS_NUMBER(y);
    double diff = fabs(x_double - y_double);
    return BOOL_VAL(diff < 1e-7);
  }
  else if (IS_STRING(x)) {
    ObjString* xString = AS_STRING(x);
    ObjString* yString = AS_STRING(y);
    if ((xString->length == yString->length) &&
        (memcmp(xString->chars, yString->chars, xString->length) == 0)) {
      return BOOL_VAL(true);
    }
    return BOOL_VAL(false);
  }
  else if (IS_LIST(x)) {
    ObjList* xList = AS_LIST(x);
    ObjList* yList = AS_LIST(y);
    if (xList->count == yList->count) {
      Value num_items = list_count(x);
      for (int i = 0; i < AS_NUMBER(num_items); i++) {
        Value xItem = list_get(x, NUMBER_VAL(i));
        Value yItem = list_get(y, NUMBER_VAL(i));
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
    size_t x_num_items = xMap->num_entries;
    size_t y_num_items = yMap->num_entries;
    if (x_num_items != y_num_items) {
      return BOOL_VAL(false);
    }
    size_t x_num_entries = xMap->num_entries;
    for (size_t i = 0; i < x_num_entries; i++) {
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

static int32_t find_indices_index(int32_t* indices, MapEntry* entries, size_t capacity, Value key) {
  /* hash the key and get an index
   * - if indices[index] is empty, return it
   * - if indices[index] points to an entry in entries with a hash that matches our hash, return index
   * Otherwise, keep adding one till we get to the correct key or an empty slot. */

  ObjString* keyString = AS_STRING(key);

  uint32_t index = keyString->hash % (uint32_t) capacity;
  for (;;) {
    if (indices[index] == MAP_EMPTY) {
      return (int32_t) index;
    }
    if (AS_BOOL(equal(entries[indices[index]].key, key))) {
      return (int32_t) index;
    }

    index = (index + 1) % (uint32_t)capacity;
  }
}

static void adjustCapacity(ObjMap* map, size_t capacity) {
  // allocate new space
  int32_t* indices = ALLOCATE(int32_t, capacity);
  MapEntry* entries = ALLOCATE(MapEntry, capacity);

  // initialize all indices to MAP_EMPTY
  for (size_t i = 0; i < capacity; i++) {
    indices[i] = MAP_EMPTY;
  }

  // copy entries over to new space, filling in indices slots as well
  size_t num_entries = map->num_entries;
  size_t entries_index = 0;
  for (; entries_index < num_entries; entries_index++) {
    // find new index
    int32_t indices_index = find_indices_index(indices, entries, capacity, map->entries[entries_index].key);
    indices[indices_index] = (int32_t) entries_index;
    entries[entries_index] = map->entries[entries_index];
  }

  // fill in remaining entries with nil values
  for (; entries_index < capacity; entries_index++) {
    entries[entries_index].hash = 0;
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
    size_t capacity = GROW_CAPACITY(map->indices_capacity);
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

  entry->hash = AS_STRING(key)->hash;
  entry->key = key;
  entry->value = value;
  return OBJ_VAL(map);
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
  ObjList* keys = allocate_list();
  size_t num_entries = map->num_entries;
  for (size_t i = 0; i < num_entries; i++) {
    list_add(OBJ_VAL(keys), map->entries[i].key);
  }
  return OBJ_VAL(keys);
}

Value map_vals(ObjMap* map) {
  ObjList* vals = allocate_list();
  size_t num_entries = map->num_entries;
  for (size_t i = 0; i < num_entries; i++) {
    list_add(OBJ_VAL(vals), map->entries[i].value);
  }
  return OBJ_VAL(vals);
}

Value map_pairs(ObjMap* map) {
  ObjList* pairs = allocate_list();
  size_t num_entries = map->num_entries;
  for (size_t i = 0; i < num_entries; i++) {
    if (!AS_BOOL(equal(map->entries[i].key, NIL_VAL))) {
      ObjList* pair = allocate_list();
      list_add(OBJ_VAL(pair), map->entries[i].key);
      list_add(OBJ_VAL(pair), map->entries[i].value);
      list_add(OBJ_VAL(pairs), OBJ_VAL(pair));
    }
  }
  return OBJ_VAL(pairs);
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
    printf("%g", AS_NUMBER(value));
  }
  else if (IS_LIST(value)) {
    Value num_items = list_count(value);
    printf("[");
    if (AS_NUMBER(num_items) > 0) {
      print(list_get(value, NUMBER_VAL(0)));
    }
    for (int i = 1; i < AS_NUMBER(num_items); i++) {
      printf(" ");
      print(list_get(value, NUMBER_VAL(i)));
    }
    printf("]");
  }
  else if (IS_MAP(value)) {
    size_t num_entries = AS_MAP(value)->num_entries;
    printf("{");
    bool first_entry = true;
    for (size_t i = 0; i < num_entries; i++) {
      if (!first_entry) {
        printf(", ");
      }
      print(AS_MAP(value)->entries[i].key);
      printf(" ");
      print(AS_MAP(value)->entries[i].value);
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

Value readline(void) {
  /* K&R p29 */
  int ch = 0;
  char buffer[MAX_LINE];
  int num_chars;
  for (num_chars=0; num_chars<(MAX_LINE-1) && (ch=getchar()) != EOF && ch != '\n'; num_chars++) {
    buffer[num_chars] = (char) ch;
  }
  if ((ch == EOF) && (num_chars == 0)) {
    return NIL_VAL;
  }
  Value result = OBJ_VAL(copyString(buffer, (size_t) num_chars));
  inc_ref(AS_OBJ(result));
  return result;
}

Value str_lower(Value string) {
  ObjString* s = AS_STRING(string);
  for (int i=0; s->chars[i] != '\0'; i++) {
    s->chars[i] = (char) tolower((int) s->chars[i]);
  }
  return string;
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

Value str_split(Value string) {
  ObjString* s = AS_STRING(string);
  ObjList* splits = allocate_list();
  size_t split_length = 0;
  int split_start_index = 0;
  for (int i=0; s->chars[i] != '\0'; i++) {
    if (s->chars[i] == ' ') {
      ObjString* split = copyString(&(s->chars[split_start_index]), split_length);
      list_add(OBJ_VAL(splits), OBJ_VAL(split));
      split_start_index = i + 1;
      split_length = 0;
    }
    else {
      split_length++;
    }
  }
  ObjString* split = copyString(&(s->chars[split_start_index]), split_length);
  list_add(OBJ_VAL(splits), OBJ_VAL(split));
  return OBJ_VAL(splits);
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
      for (size_t i = 0; i < list->count; i++) {
        Value v = list_get(OBJ_VAL(object), NUMBER_VAL((double)i));
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
      for (size_t i = 0; i < map->num_entries; i++) {
        MapEntry entry = map->entries[i];
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

Value let(ObjMap* user_globals, Value words, Value i, Value word, Value recur, Value counts, Value numwords) {
    Value user_global_lookup_1 = OBJ_VAL(copyString("counts", (size_t) 6));
  inc_ref(AS_OBJ(user_global_lookup_1));
  Value curcount = map_get(AS_MAP(map_get(user_globals, user_global_lookup_1, NIL_VAL)), word, NUMBER_VAL(0));

  Value user_global_lookup_2 = OBJ_VAL(copyString("counts", (size_t) 6));
  inc_ref(AS_OBJ(user_global_lookup_2));
  Value map_assoc = map_set(AS_MAP(map_get(user_globals, user_global_lookup_2, NIL_VAL)), word, add(curcount, NUMBER_VAL(1)));

  if (IS_OBJ(map_assoc)) {
    inc_ref(AS_OBJ(map_assoc));
  }
  dec_ref_and_free(AS_OBJ(user_global_lookup_1));
  dec_ref_and_free(AS_OBJ(user_global_lookup_2));
  return map_assoc;
}

Value if_form(ObjMap* user_globals, Value words, Value i, Value word, Value recur, Value counts, Value numwords) {

  Value str_blank_ = str_blank(word);  if (AS_BOOL(equal(BOOL_VAL(false), str_blank_))) {
  Value let_result = let(user_globals, words, i, word, recur, counts, numwords);
  if (IS_OBJ(let_result)) {
    inc_ref(AS_OBJ(let_result));
  }
  if (IS_OBJ(let_result)) {
  dec_ref_and_free(AS_OBJ(let_result));
  }
    return let_result;
  }

  else {
    return NIL_VAL;
  }
}

Value do_f(ObjMap* user_globals, Value words, Value i, Value word, Value recur, Value counts, Value numwords) {
  Value if_form_result = if_form(user_globals, words, i, word, recur, counts, numwords);
  recur_add(AS_RECUR(recur), add(i, NUMBER_VAL(1)));
  if (IS_OBJ(if_form_result)) {
    dec_ref_and_free(AS_OBJ(if_form_result));
  }
  return recur;
}

Value let_1(ObjMap* user_globals, Value words, Value i, Value recur, Value counts, Value numwords) {
    Value word = list_get(words, i);

  Value do_result = do_f(user_globals, words, i, word, recur, counts, numwords);

  if (IS_OBJ(do_result)) {
    inc_ref(AS_OBJ(do_result));
  }
  if (IS_OBJ(do_result)) {
  dec_ref_and_free(AS_OBJ(do_result));
  }
  return do_result;
}

Value if_form_1(ObjMap* user_globals, Value words, Value i, Value recur, Value counts, Value numwords) {

  if (AS_BOOL(equal(i, numwords))) {
  Value user_global_lookup = OBJ_VAL(copyString("counts", (size_t) 6));
  inc_ref(AS_OBJ(user_global_lookup));
  if (IS_OBJ(map_get(user_globals, user_global_lookup, NIL_VAL))) {
    inc_ref(AS_OBJ(map_get(user_globals, user_global_lookup, NIL_VAL)));
  }
  dec_ref_and_free(AS_OBJ(user_global_lookup));
    return map_get(user_globals, user_global_lookup, NIL_VAL);
  }
  else {
  Value let_result = let_1(user_globals, words, i, recur, counts, numwords);
  if (IS_OBJ(let_result)) {
    inc_ref(AS_OBJ(let_result));
  }
  if (IS_OBJ(let_result)) {
  dec_ref_and_free(AS_OBJ(let_result));
  }
    return let_result;
  }
}

Value loop(ObjMap* user_globals, Value words, Value counts, Value numwords) {
  Recur recur_1;
  recur_init(&recur_1);
  Value recur = RECUR_VAL(&recur_1);
  Value i = NUMBER_VAL(0);
  if (IS_OBJ(i)) {
    inc_ref(AS_OBJ(i));
  }
  bool continueFlag = false;
  do {
  Value if_form_result = if_form_1(user_globals, words, i, recur, counts, numwords);
  Value result = if_form_result;
  if (IS_RECUR(result)) {
    /* grab values from result and update  */
    if (IS_OBJ(i)) {
      dec_ref_and_free(AS_OBJ(i));
    }
    i = recur_get(result, NUMBER_VAL(0));
    if (IS_OBJ(i)) {
      inc_ref(AS_OBJ(i));
    }
    continueFlag = true;
    recur_free(&recur_1);
  }
  else {

  if (IS_OBJ(result)) {
    inc_ref(AS_OBJ(result));
  }
    recur_free(&recur_1);
    return result;
  }
  } while (continueFlag);
  return NIL_VAL;
}

Value let_2(ObjMap* user_globals, Value words, Value counts) {
    Value numwords = list_count(words);

  Value loop_result = loop(user_globals, words,counts,numwords);

  if (IS_OBJ(loop_result)) {
    inc_ref(AS_OBJ(loop_result));
  }
  if (IS_OBJ(loop_result)) {
    dec_ref_and_free(AS_OBJ(loop_result));
  }
  return loop_result;
}

Value u_process_M_words(ObjMap* user_globals, Value counts, Value words) {
    Value let_result = let_2(user_globals, words, counts);
  Value result = let_result;
  if (IS_OBJ(result)) {
    inc_ref(AS_OBJ(result));
  }
  if (IS_OBJ(let_result)) {
  dec_ref_and_free(AS_OBJ(let_result));
  }
  return result;
}

Value let_3(ObjMap* user_globals, Value counts, Value line) {
    Value str_lower_ = str_lower(line);
  Value str_split_ = str_split(str_lower_);
  inc_ref(AS_OBJ(str_split_));
  Value words = str_split_;

  Value user_global_lookup_3 = OBJ_VAL(copyString("counts", (size_t) 6));
  inc_ref(AS_OBJ(user_global_lookup_3));
  Value u_f_result = u_process_M_words(user_globals, map_get(user_globals, user_global_lookup_3, NIL_VAL), words);

  if (IS_OBJ(u_f_result)) {
    inc_ref(AS_OBJ(u_f_result));
  }
  dec_ref_and_free(AS_OBJ(str_split_));
  dec_ref_and_free(AS_OBJ(user_global_lookup_3));
  if (IS_OBJ(u_f_result)) {
    dec_ref_and_free(AS_OBJ(u_f_result));
  }
  return u_f_result;
}

Value u_process_M_line(ObjMap* user_globals, Value counts, Value line) {
    Value let_result = let_3(user_globals, counts, line);
  Value result = let_result;
  if (IS_OBJ(result)) {
    inc_ref(AS_OBJ(result));
  }
  if (IS_OBJ(let_result)) {
  dec_ref_and_free(AS_OBJ(let_result));
  }
  return result;
}

Value u_compare(ObjMap* user_globals, Value a, Value b) {
  
  Value result = greater(user_globals, list_get(a, NUMBER_VAL(1)), list_get(b, NUMBER_VAL(1)));
  if (IS_OBJ(result)) {
    inc_ref(AS_OBJ(result));
  }

  return result;
}

Value if_form_2(ObjMap* user_globals, Value recur, Value line) {

  Value str_blank_ = str_blank(line);  if (AS_BOOL(equal(BOOL_VAL(false), str_blank_))) {
  Value user_global_lookup_4 = OBJ_VAL(copyString("counts", (size_t) 6));
  inc_ref(AS_OBJ(user_global_lookup_4));
  Value u_f_result = u_process_M_line(user_globals, map_get(user_globals, user_global_lookup_4, NIL_VAL), line);
  if (IS_OBJ(u_f_result)) {
    inc_ref(AS_OBJ(u_f_result));
  }
  dec_ref_and_free(AS_OBJ(user_global_lookup_4));
  if (IS_OBJ(u_f_result)) {
    dec_ref_and_free(AS_OBJ(u_f_result));
  }
    return u_f_result;
  }

  else {
    return NIL_VAL;
  }
}

Value do_f_1(ObjMap* user_globals, Value recur, Value line) {
  Value if_form_result = if_form_2(user_globals, recur, line);
  Value readline_result = readline();
  recur_add(AS_RECUR(recur), readline_result);
  if (IS_OBJ(if_form_result)) {
    dec_ref_and_free(AS_OBJ(if_form_result));
  }
  if (IS_OBJ(readline_result)) {
    dec_ref_and_free(AS_OBJ(readline_result));
  }
  return recur;
}

Value if_form_3(ObjMap* user_globals, Value recur, Value line) {

  if (AS_BOOL(equal(NIL_VAL, line))) {

  if (IS_OBJ(NIL_VAL)) {
    inc_ref(AS_OBJ(NIL_VAL));
  }

    return NIL_VAL;
  }
  else {
  Value do_result = do_f_1(user_globals, recur, line);
  if (IS_OBJ(do_result)) {
    inc_ref(AS_OBJ(do_result));
  }
  if (IS_OBJ(do_result)) {
  dec_ref_and_free(AS_OBJ(do_result));
  }
    return do_result;
  }
}

Value loop_1(ObjMap* user_globals) {
  Recur recur_1;
  recur_init(&recur_1);
  Value recur = RECUR_VAL(&recur_1);
  Value readline_result = readline();
  Value line = readline_result;
  if (IS_OBJ(line)) {
    inc_ref(AS_OBJ(line));
  }
  bool continueFlag = false;
  do {
  Value if_form_result = if_form_3(user_globals, recur, line);
  Value result = if_form_result;
  if (IS_RECUR(result)) {
    /* grab values from result and update  */
    if (IS_OBJ(line)) {
      dec_ref_and_free(AS_OBJ(line));
    }
    line = recur_get(result, NUMBER_VAL(0));
    if (IS_OBJ(line)) {
      inc_ref(AS_OBJ(line));
    }
    continueFlag = true;
    recur_free(&recur_1);
  }
  else {
  if (IS_OBJ(readline_result)) {
    dec_ref_and_free(AS_OBJ(readline_result));
  }
  if (IS_OBJ(result)) {
    inc_ref(AS_OBJ(result));
  }
    recur_free(&recur_1);
    return result;
  }
  } while (continueFlag);
  return NIL_VAL;
}

Value do_f_2(ObjMap* user_globals, Value entry, Value numitems, Value index, Value sortedlist, Value recur) {
  Value print_result = print(list_get(entry, NUMBER_VAL(0)));
  Value str = OBJ_VAL(copyString(" ", (size_t) 1));
  inc_ref(AS_OBJ(str));
  Value print_result_1 = print(str);
  Value print_result_2 = print(list_get(entry, NUMBER_VAL(1)));
  Value str_1 = OBJ_VAL(copyString("\n", (size_t) 2));
  inc_ref(AS_OBJ(str_1));
  Value print_result_3 = print(str_1);
  recur_add(AS_RECUR(recur), add(index, NUMBER_VAL(1)));
  dec_ref_and_free(AS_OBJ(str));
  dec_ref_and_free(AS_OBJ(str_1));
  return recur;
}

Value let_4(ObjMap* user_globals, Value index, Value recur, Value numitems, Value sortedlist) {
    Value entry = list_get(sortedlist, index);

  Value do_result = do_f_2(user_globals, entry, numitems, index, sortedlist, recur);

  if (IS_OBJ(do_result)) {
    inc_ref(AS_OBJ(do_result));
  }
  if (IS_OBJ(do_result)) {
  dec_ref_and_free(AS_OBJ(do_result));
  }
  return do_result;
}

Value if_form_4(ObjMap* user_globals, Value index, Value recur, Value numitems, Value sortedlist) {

  if (AS_BOOL(less(user_globals, index, numitems))) {
  Value let_result = let_4(user_globals, index, recur, numitems, sortedlist);
  if (IS_OBJ(let_result)) {
    inc_ref(AS_OBJ(let_result));
  }
  if (IS_OBJ(let_result)) {
  dec_ref_and_free(AS_OBJ(let_result));
  }
    return let_result;
  }

  else {
    return NIL_VAL;
  }
}

Value loop_2(ObjMap* user_globals, Value numitems, Value sortedlist) {
  Recur recur_1;
  recur_init(&recur_1);
  Value recur = RECUR_VAL(&recur_1);
  Value index = NUMBER_VAL(0);
  if (IS_OBJ(index)) {
    inc_ref(AS_OBJ(index));
  }
  bool continueFlag = false;
  do {
  Value if_form_result = if_form_4(user_globals, index, recur, numitems, sortedlist);
  Value result = if_form_result;
  if (IS_RECUR(result)) {
    /* grab values from result and update  */
    if (IS_OBJ(index)) {
      dec_ref_and_free(AS_OBJ(index));
    }
    index = recur_get(result, NUMBER_VAL(0));
    if (IS_OBJ(index)) {
      inc_ref(AS_OBJ(index));
    }
    continueFlag = true;
    recur_free(&recur_1);
  }
  else {

  if (IS_OBJ(result)) {
    inc_ref(AS_OBJ(result));
  }
    recur_free(&recur_1);
    return result;
  }
  } while (continueFlag);
  return NIL_VAL;
}

Value let_5(ObjMap* user_globals) {
    Value user_global_lookup_5 = OBJ_VAL(copyString("counts", (size_t) 6));
  inc_ref(AS_OBJ(user_global_lookup_5));
  Value map_pairs_ = map_pairs(AS_MAP(map_get(user_globals, user_global_lookup_5, NIL_VAL)));
  inc_ref(AS_OBJ(map_pairs_));
  Value sortedlist = list_sort(user_globals, map_pairs_, u_compare);

  Value numitems = list_count(sortedlist);

  Value loop_result = loop_2(user_globals, numitems,sortedlist);

  if (IS_OBJ(loop_result)) {
    inc_ref(AS_OBJ(loop_result));
  }
  dec_ref_and_free(AS_OBJ(user_global_lookup_5));
  dec_ref_and_free(AS_OBJ(map_pairs_));
  if (IS_OBJ(loop_result)) {
    dec_ref_and_free(AS_OBJ(loop_result));
  }
  return loop_result;
}

int main(void)
{
  ObjMap* user_globals = allocate_map();
  Value map = OBJ_VAL(allocate_map());
  inc_ref(AS_OBJ(map));

  map_set(user_globals, OBJ_VAL(copyString("counts", (size_t) 6)), map);

  Value loop_result = loop_1(user_globals);
  Value let_result = let_5(user_globals);
  dec_ref_and_free(AS_OBJ(map));
  if (IS_OBJ(loop_result)) {
    dec_ref_and_free(AS_OBJ(loop_result));
  }
  if (IS_OBJ(let_result)) {
  dec_ref_and_free(AS_OBJ(let_result));
  }
  free_object((Obj*)user_globals);
  return 0;
}