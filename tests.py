import os
from os import linesep as LSEP
import platform
import subprocess
import tempfile
import unittest
import sys
from unittest.mock import patch
from lang import TokenType, scan_tokens, parse, Symbol, _compile, _get_compile_cmd_env
from lang import (
        GCC_CMD, GCC_CHECK_OPTIONS, GCC_CHECK_ENV,
        CLANG_CMD, CLANG_CHECK_OPTIONS, CLANG_CHECK_ENV,
    )


SAVE_FAILED = False
QUICK = False
CI = False


SOURCE = '(+ 10 2 (- 15 (+ 4 4)) -5)'
EXPECTED_TOKENS = [
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '+'},
    {'type': TokenType.NUMBER, 'lexeme': '10'},
    {'type': TokenType.NUMBER, 'lexeme': '2'},
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '-'},
    {'type': TokenType.NUMBER, 'lexeme': '15'},
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '+'},
    {'type': TokenType.NUMBER, 'lexeme': '4'},
    {'type': TokenType.NUMBER, 'lexeme': '4'},
    {'type': TokenType.RIGHT_PAREN},
    {'type': TokenType.RIGHT_PAREN},
    {'type': TokenType.NUMBER, 'lexeme': '-5'},
    {'type': TokenType.RIGHT_PAREN},
]

EXPECTED_AST_FORMS = [
    [
        Symbol('+'),
        10,
        2,
        [
            Symbol('-'),
            15,
            [
                Symbol('+'),
                4,
                4,
            ]
        ],
        -5,
    ]
]


class ScanTokenTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        tokens = scan_tokens(SOURCE)
        self.assertEqual(tokens, EXPECTED_TOKENS)

    def test_nil(self):
        tokens = scan_tokens('(= 1 nil)')
        self.assertEqual(tokens,
                [
                    {'type': TokenType.LEFT_PAREN},
                    {'type': TokenType.SYMBOL, 'lexeme': '='},
                    {'type': TokenType.NUMBER, 'lexeme': '1'},
                    {'type': TokenType.NIL},
                    {'type': TokenType.RIGHT_PAREN},
                ]
            )

    def test_number(self):
        tokens = scan_tokens('2')
        self.assertEqual(tokens, [{'type': TokenType.NUMBER, 'lexeme': '2'}])


class ParseTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        ast = parse(EXPECTED_TOKENS)
        self.assertEqual(ast.forms, EXPECTED_AST_FORMS)

    def test_2(self):
        tokens = [
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': '='},
            {'type': TokenType.TRUE},
            {'type': TokenType.NIL},
            {'type': TokenType.RIGHT_PAREN},
        ]
        ast = parse(tokens)
        self.assertEqual(ast.forms,
            [
                [
                    Symbol('='),
                    True,
                    None
                ]
            ]
        )

    def test_sequence_of_forms(self):
        tokens = [
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': 'def'},
            {'type': TokenType.SYMBOL, 'lexeme': 'a'},
            {'type': TokenType.NUMBER, 'lexeme': '1'},
            {'type': TokenType.RIGHT_PAREN},
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': '+'},
            {'type': TokenType.SYMBOL, 'lexeme': 'a'},
            {'type': TokenType.NUMBER, 'lexeme': '1'},
            {'type': TokenType.RIGHT_PAREN},
        ]
        ast = parse(tokens)
        self.assertEqual(ast.forms,
                [
                    [
                        Symbol('def'),
                        Symbol('a'),
                        1
                    ],
                    [
                        Symbol('+'),
                        Symbol('a'),
                        1
                    ]
                ]
            )


gcc_cmd = os.environ.get('GCC', GCC_CMD)
clang_cmd = os.environ.get('CLANG', CLANG_CMD)


def _build_sqlite(cc_cmd):
    compile_cmd = cc_cmd + ['--shared', '-fPIC', '-o', 'libsqlite3.so', os.path.join('lib', 'sqlite3.c')]
    try:
        subprocess.run(compile_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f'error building libsqlite.so: {e.stderr.decode("utf8")}')


def _run_test(test, assert_equal, sqlite=False):
    print(f'*** c test: {test["src"]}')
    c_code = _compile(test['src'])

    if CI:
        if platform.system() == 'Darwin':
            compilers = [ (gcc_cmd, False, 'gcc_regular') ]
        elif platform.system() == 'Windows':
            cc_path = 'clang.exe'
            compilers = [ (cc_path, False, 'clang_regular') ]
        else:
            if sqlite:
                compilers = [ (gcc_cmd, False, 'gcc_regular') ]
            else:
                compilers = [
                    (clang_cmd, True, 'clang_checks'),
                    (gcc_cmd, True, 'gcc_checks'),
                ]
    else:
        compilers = [ (clang_cmd, False, 'clang_regular') ]
        if not QUICK:
            compilers.extend([
                (clang_cmd, True, 'clang_checks'),
                (gcc_cmd, True, 'gcc_checks'),
            ])

    with tempfile.TemporaryDirectory() as tmp:
        c_filename = os.path.join(tmp, 'code.c')
        with open(c_filename, 'wb') as f:
            f.write(c_code.encode('utf8'))

        custom_code = c_code.split('/* CUSTOM CODE */\n\n')[-1]

        for cc_cmd, with_checks, env_name in compilers:
            print(f'  ({env_name})')
            program_filename = os.path.join(tmp, env_name)

            compile_cmd, env = _get_compile_cmd_env(file_name=c_filename, output_file_name=program_filename, with_checks=with_checks)
            try:
                subprocess.run(compile_cmd, check=True, env=env, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f'compile_cmd: {compile_cmd}')
                print(f'bad c code:\n{custom_code}')
                print(f'err: {e.stderr.decode("utf8")}')
                if SAVE_FAILED:
                    with tempfile.NamedTemporaryFile(delete=False, dir='.', suffix='.c') as f:
                        f.write(c_code.encode('utf8'))
                raise

            program_cmd = [program_filename]
            if 'input' in test:
                input_ = test['input'].encode('utf8')
            else:
                input_ = None

            try:
                result = subprocess.run(program_cmd, check=True, input=input_, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f'bad c code:\n{custom_code}')
                print(f'err: {e.stderr.decode("utf8")}')
                print(f'out: {e.stdout.decode("utf8")}')
                if SAVE_FAILED:
                    with tempfile.NamedTemporaryFile(delete=False, dir='.', suffix='.c') as f:
                        f.write(c_code.encode('utf8'))
                raise

            try:
                assert_equal(result.stdout.decode('utf8'), test['output'])
            except AssertionError:
                print(f'bad c code:\n{custom_code}')
                if SAVE_FAILED:
                    with tempfile.NamedTemporaryFile(delete=False, dir='.', suffix='.c') as f:
                        f.write(c_code.encode('utf8'))
                raise


class CompileTests(unittest.TestCase):
    def test_values(self):
        tests = [
            {'src': '(print nil)', 'output': 'nil'},
            {'src': '(print true)', 'output': 'true'},
            {'src': '(print false)', 'output': 'false'},
            {'src': '(print 1)', 'output': '1'},
            {'src': '(print -1)', 'output': '-1'},
            {'src': '(print 1.2)', 'output': '1.2'},
            {'src': '(print -1.2)', 'output': '-1.2'},
            {'src': '(print 1/10)', 'output': '1/10'},
            {'src': '(print -1/10)', 'output': '-1/10'},
            {'src': '(print 1/-10)', 'output': '1/-10'},
            {'src': '(print 1/-10)', 'output': '1/-10'},
            {'src': '(print -1/-10)', 'output': '-1/-10'},
            {'src': '(print 5/10)', 'output': '1/2'},
            {'src': '(print 15/50)', 'output': '3/10'},
            {'src': '(print (nil? 0))', 'output': 'false'},
            {'src': '(print (nil? nil))', 'output': 'true'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_math_operations(self):
        tests = [
            {'src': '(print (+ 1 3))', 'output': '4'},
            {'src': '(print (+ 1.5 2.3))', 'output': '3.8'},
            {'src': '(print (+ 1 3 2))', 'output': '6'},
            {'src': '(print (+ 1/10 2/10))', 'output': '3/10'},
            {'src': '(print (+ 1/10 1/5))', 'output': '3/10'},
            {'src': '(print (+ 1/10 2/10 2/10))', 'output': '1/2'},
            {'src': '(print (- 3 2))', 'output': '1'},
            {'src': '(print (- 3.5 2.1))', 'output': '1.4'},
            {'src': '(print (- 6 3 1))', 'output': '2'},
            {'src': '(print (* 3 2))', 'output': '6'},
            {'src': '(print (* 3.6 2.5))', 'output': '9'},
            {'src': '(print (* 3 2 2))', 'output': '12'},
            {'src': '(print (/ 6 2))', 'output': '3'},
            {'src': '(print (/ 7.5 2.5))', 'output': '3'},
            {'src': '(print (/ 12 2 3))', 'output': '2'},
            {'src': '(print (+ 1 "a"))', 'output': 'ERROR: Type'},
            {'src': '(print (+ 1 "a" 2))', 'output': 'ERROR: Type'},
            {'src': '(print (+ "a" 1 2))', 'output': 'ERROR: Type'},
            {'src': '(print (- "a" 1))', 'output': 'ERROR: Type'},
            {'src': '(print (- 1 "a" 2))', 'output': 'ERROR: Type'},
            {'src': '(print (- "a" 1 2))', 'output': 'ERROR: Type'},
            {'src': '(print (* 1 "a"))', 'output': 'ERROR: Type'},
            {'src': '(print (* 1 "a" 2))', 'output': 'ERROR: Type'},
            {'src': '(print (* "a" 1 2))', 'output': 'ERROR: Type'},
            {'src': '(print (/ "a" 1))', 'output': 'ERROR: Type'},
            {'src': '(print (/ 1 "a" 2))', 'output': 'ERROR: Type'},
            {'src': '(print (/ "a" 1 2))', 'output': 'ERROR: Type'},
            {'src': '(print (/ 1 0))', 'output': 'ERROR: DivideByZero'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_core_strings(self):
        tests = [
            {'src': '(print "abc")', 'output': 'abc'},
            {'src': '(print "abcdefgh")', 'output': 'abcdefgh'},
            {'src': '(print (count "abc"))', 'output': '3'},
            {'src': '(print (count "abcdefgh"))', 'output': '8'},
            {'src': '(print (str))', 'output': ''},
            {'src': '(print (str nil))', 'output': ''},
            {'src': '(print (str true))', 'output': 'true'},
            {'src': '(print (str false))', 'output': 'false'},
            {'src': '(print (str 1))', 'output': '1'},
            {'src': '(print (str "abc"))', 'output': 'abc'},
            {'src': '(print (str "abcdefgh"))', 'output': 'abcdefgh'},
            {'src': '(print (str "abc" "def"))', 'output': 'abcdef'},
            {'src': '(print (str "abcd" "efgh"))', 'output': 'abcdefgh'},
            {'src': '(print (str "abcdefgh" "another1"))', 'output': 'abcdefghanother1'},
            {'src': '(print (str "abc" "abc"))', 'output': 'abcabc'},
            {'src': '(print (str "abcd" "abcd"))', 'output': 'abcdabcd'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_lists(self):
        tests = [
            {'src': '(print [])', 'output': '[]'},
            {'src': '(print [1])', 'output': '[1]'},
            {'src': '(print [1 nil "hello" 2.34 true [1] {"a" 1}])', 'output': '[1 nil hello 2.34 true [1] {a 1}]'},
            {'src': '(print (nth [1 2] 0))', 'output': '1'},
            {'src': '(print (nth ["1" 2] 0))', 'output': '1'},
            {'src': '(print (nth [1 (+ 1 1)] 1))', 'output': '2'},
            {'src': '(print (nth [1 (nth [2 3] 0)] 1))', 'output': '2'},
            {'src': '(print (nth [1 nil 2] 1))', 'output': 'nil'},
            {'src': '(print (nth [1 2 3] 3))', 'output': 'nil'},
            {'src': '(print (nth [1 2 3] -1))', 'output': 'nil'},
            {'src': '(print (count [1 2 3]))', 'output': '3'},
            {'src': '(print (conj nil 0))', 'output': 'ERROR: Type'},
            {'src': '(print (conj [] 0))', 'output': '[0]'},
            {'src': '(print (conj [1 2 3 4] "a"))', 'output': '[1 2 3 4 a]'},
            {'src': '(print (remove [1 2 3] 0))', 'output': '[2 3]'},
            {'src': '(print (remove [1 2 3] 1))', 'output': '[1 3]'},
            {'src': '(print (remove [1 2 3] 2))', 'output': '[1 2]'},
            {'src': '(print (sort [1]))', 'output': '[1]'},
            {'src': '(print (sort [1 2]))', 'output': '[1 2]'},
            {'src': '(print (sort [1 3 2]))', 'output': '[1 2 3]'},
            {'src': '(print (sort > [1 3 2]))', 'output': '[3 2 1]'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_hash(self):
        tests = [
            {'src': '(print (hash nil))', 'output': '84696351'},
            {'src': '(print (hash false))', 'output': '67918732'},
            {'src': '(print (hash true))', 'output': '118251589'},
            {'src': '(print (hash 1))', 'output': '70638592'},
            {'src': '(print (hash 2))', 'output': '120971449'},
            {'src': '(print (hash "abc"))', 'output': '4189669961'},
            {'src': '(print (hash "def"))', 'output': '3033789270'},
            {'src': '(print (hash "abcdef"))', 'output': '3728255472'},
            {'src': '(print (hash "abcdefgh"))', 'output': '670277762'},
            {'src': '(print (hash "123"))', 'output': '1040253977'},
            {'src': '(print (hash "1"))', 'output': '274927234'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_maps(self):
        tests = [
            {'src': '(print {})', 'output': '{}'},
            {'src': '(print {nil "a"})', 'output': '{nil a}'},
            {'src': '(print {true "a"})', 'output': '{true a}'},
            {'src': '(print {false "a"})', 'output': '{false a}'},
            {'src': '(print {1 nil})', 'output': '{1 nil}'},
            {'src': '(print {"a" nil})', 'output': '{a nil}'},
            {'src': '(print {"a" true})', 'output': '{a true}'},
            {'src': '(print {"a" false})', 'output': '{a false}'},
            {'src': '(print {"a" 1})', 'output': '{a 1}'},
            {'src': '(print {"a" "1"})', 'output': '{a 1}'},
            {'src': '(print {"a" {"1" "value"}})', 'output': '{a {1 value}}'},
            {'src': '(print {"a" 1 "b" 2})', 'output': '{a 1, b 2}'},
            {'src': '(print {"a" [1] "b" [2]})', 'output': '{a [1], b [2]}'},
            {'src': '(print {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7})',
                 'output': '{1 1, 2 2, 3 3, 4 4, 5 5, 6 6, 7 7}'},
            {'src': '(print (keys {"a" 2 "b" 3}))', 'output': '[a b]'},
            {'src': '(print (vals {"a" 2 "b" 3}))', 'output': '[2 3]'},
            {'src': '(print (pairs {"a" 2 "b" 3}))', 'output': '[[a 2] [b 3]]'},
            {'src': '(print (assoc {} "new-key" "new-val"))', 'output': '{new-key new-val}'},
            {'src': '(print (assoc {"1" 2 "a" 3} "new-key" "new-val"))', 'output': '{1 2, a 3, new-key new-val}'},
            {'src': '(print (assoc {"1" 2 "a" 3} "1" "new-val"))', 'output': '{1 new-val, a 3}'},
            {'src': '(print (assoc {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7 "8" 8} "9" 9))',
                'output': '{1 1, 2 2, 3 3, 4 4, 5 5, 6 6, 7 7, 8 8, 9 9}'},
            {'src': '(print (get {} "a"))', 'output': 'nil'},
            {'src': '(print (get {} "a" 99))', 'output': '99'},
            {'src': '(print (get {"a" 1} "a"))', 'output': '1'},
            {'src': '(print (get {"a" 1} "b"))', 'output': 'nil'},
            {'src': '(print (get {"a" 1} "b" 99))', 'output': '99'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5} "1"))', 'output': '1'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6} "1"))', 'output': '1'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7} "1"))', 'output': '1'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7 "8" 8 "9" 9} "1"))', 'output': '1'},
            {'src': '(print (get {"1" {}} "1"))', 'output': '{}'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7 "8" 8 "9" 9} "a"))', 'output': 'nil'},
            {'src': '(print (contains? {} "a"))', 'output': 'false'},
            {'src': '(print (contains? {"a" 1} 1))', 'output': 'false'},
            {'src': '(print (contains? {"a" 1} "a"))', 'output': 'true'},
            {'src': '(print (contains? {"a" 1} "z"))', 'output': 'false'},
            {'src': '(print (contains? {1 2} 1))', 'output': 'true'},
            {'src': '(print (contains? {1 2} 2))', 'output': 'false'},
            {'src': '(print (contains? {nil 2} nil))', 'output': 'true'},
            {'src': '(print (contains? {true 2} true))', 'output': 'true'},
            {'src': '(print (contains? {false 2} false))', 'output': 'true'},
            {'src': '(print (dissoc {} "1"))', 'output': '{}'},
            {'src': '(print (dissoc {"1" 1} "1"))', 'output': '{}'},
            {'src': '(print (dissoc {"1" 1 "2" 2} "2"))', 'output': '{1 1}'},
            {'src': '(print (dissoc {"1" 1 "2" 2} "1"))', 'output': '{2 2}'},
            {'src': '(print (dissoc {"abcdefg" 1 "anotherkey" 2} "abcdefg"))', 'output': '{anotherkey 2}'},
            {'src': '(print (dissoc {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5} "7"))',
                'output': '{1 1, 2 2, 3 3, 4 4, 5 5}'},
            {'src': '(print (dissoc {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6} "7"))',
                'output': '{1 1, 2 2, 3 3, 4 4, 5 5, 6 6}'},
            {'src': '(print (dissoc {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7} "3"))',
                'output': '{1 1, 2 2, 4 4, 5 5, 6 6, 7 7}'},
            {'src': '(print (dissoc {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7} "8"))',
                'output': '{1 1, 2 2, 3 3, 4 4, 5 5, 6 6, 7 7}'},
            {'src': '(print (dissoc {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7 "8" 8} "8"))',
                'output': '{1 1, 2 2, 3 3, 4 4, 5 5, 6 6, 7 7}'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_convert(self):
        tests = [
            {'src': '(print (to-number nil))', 'output': 'ERROR: Type'},
            {'src': '(print (to-number 4))', 'output': '4'},
            {'src': '(print (to-number "4"))', 'output': '4'},
            {'src': '(print (to-number "1111111"))', 'output': '1111111'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_comparisons(self):
        tests = [
            {'src': '(print (= nil nil))', 'output': 'true'},
            {'src': '(print (= true true))', 'output': 'true'},
            {'src': '(print (= false false))', 'output': 'true'},
            {'src': '(print (= nil false))', 'output': 'false'},
            {'src': '(print (= nil true))', 'output': 'false'},
            {'src': '(print (= false true))', 'output': 'false'},
            {'src': '(print (= 1 1))', 'output': 'true'},
            {'src': '(print (= 1 2))', 'output': 'false'},
            {'src': '(print (= 1 "1"))', 'output': 'false'},
            {'src': '(print (= 1 1.0))', 'output': 'true'}, #different from clojure
            {'src': '(print (= "abc" "abc"))', 'output': 'true'},
            {'src': '(print (= "abcdefgh" "abcdefgh"))', 'output': 'true'},
            {'src': '(print (= "abc" "def"))', 'output': 'false'},
            {'src': '(print (= "abcdefgh" "jklmnopq"))', 'output': 'false'},
            {'src': '(print (= [] []))', 'output': 'true'},
            {'src': '(print (= [1] [2]))', 'output': 'false'},
            {'src': '(print (= [1] [1]))', 'output': 'true'},
            {'src': '(print (= ["a" 1] ["a" 1]))', 'output': 'true'},
            {'src': '(print (= {} {}))', 'output': 'true'},
            {'src': '(print (= {"a" 1} {"a" 1}))', 'output': 'true'},
            {'src': '(print (= {"a" 2} {"a" 1}))', 'output': 'false'},
            {'src': '(print (= {"a" 1} {"b" 1}))', 'output': 'false'},
            {'src': '(print (> 3 2))', 'output': 'true'},
            {'src': '(print (> 3 4))', 'output': 'false'},
            {'src': '(print (>= 3 3))', 'output': 'true'},
            {'src': '(print (>= 3 4))', 'output': 'false'},
            {'src': '(print (< 2 3))', 'output': 'true'},
            {'src': '(print (< 2 1))', 'output': 'false'},
            {'src': '(print (<= 2 2))', 'output': 'true'},
            {'src': '(print (<= 2 1))', 'output': 'false'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_not_or_and(self):
        tests = [
            {'src': '(print (not nil))', 'output': 'true'},
            {'src': '(print (not true))', 'output': 'false'},
            {'src': '(print (not false))', 'output': 'true'},
            {'src': '(print (not []))', 'output': 'false'},
            {'src': '(print (not 0))', 'output': 'false'},
            {'src': '(print (not {}))', 'output': 'false'},
            {'src': '(print (and nil))', 'output': 'nil'},
            {'src': '(print (and false))', 'output': 'false'},
            {'src': '(print (and 1))', 'output': '1'},
            {'src': '(print (and "1" false 2))', 'output': 'false'},
            {'src': '(print (and 1 {} 3))', 'output': '3'},
            {'src': '(print (or nil))', 'output': 'nil'},
            {'src': '(print (or false))', 'output': 'false'},
            {'src': '(print (or "1"))', 'output': '1'},
            {'src': '(print (or false "1" nil))', 'output': '1'},
            {'src': '(print (or false nil))', 'output': 'nil'},
            {'src': '(print (or 1 "2" 3))', 'output': '1'},
            {'src': '(print (or "a" 2 nil))', 'output': 'a'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_if(self):
        tests = [
            {'src': '(print (if true true))', 'output': 'true'},
            {'src': '(print (if true true false))', 'output': 'true'},
            {'src': '(print (if true "True" false))', 'output': 'True'},
            {'src': '(print (if false true))', 'output': 'nil'},
            {'src': '(print (if false true false))', 'output': 'false'},
            {'src': '(print (if false true "False"))', 'output': 'False'},
            {'src': '(print (if true ["True"] false))', 'output': '[True]'},
            {'src': '(print (if true {"True" "1"} false))', 'output': '{True 1}'},
            {'src': '(print (if 0 {"True" "1"} false))', 'output': '{True 1}'},
            {'src': '(print (if {} {"True" "1"} false))', 'output': '{True 1}'},
            {'src': '(print (if false true ["False"]))', 'output': '[False]'},
            {'src': '(print (if false true {"False" "1"}))', 'output': '{False 1}'},
            {'src': '(print (if nil true {"False" "1"}))', 'output': '{False 1}'},
            {'src': '(print (if (not false) "truthy" "falsey")', 'output': 'truthy'},
            {'src': '(print (if (not true) "truthy" "falsey")', 'output': 'falsey'},
            {'src': '(print (if (not []) "truthy" "falsey")', 'output': 'falsey'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_do(self):
        tests = [
            {'src': '(do (print 1) (print 2))', 'output': '12'},
            {'src': '(do (print 1) (if (< 1 2) (print 1) (print 2)))', 'output': '11'},
            {'src': '(print (do (print 1) (if (< 1 2) (print 1) (print 2)) "3"))', 'output': '113'},
            {'src': '(do (println "line1") (println "line2"))', 'output': f'line1{LSEP}line2{LSEP}'},
            {'src': '(print (do (println "output") 2))', 'output': f'output{LSEP}2'},
            {'src': '(print (do (println "output") "return"))', 'output': f'output{LSEP}return'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_let(self):
        tests = [
            {'src': '(print (let [y 1] y))', 'output': '1'},
            {'src': '(print (let [y "1"] y))', 'output': '1'},
            {'src': '(print (let [y 1] [y]))', 'output': '[1]'},
            {'src': '(print (let [y "1"] [y]))', 'output': '[1]'},
            {'src': '(print (let [y "1"] {"a" y}))', 'output': '{a 1}'},
            {'src': '(let [x 1] (print x))', 'output': '1'},
            {'src': '(let [x "1"] (print x))', 'output': '1'},
            {'src': '(let [x {}] (print x))', 'output': '{}'},
            {'src': '(let [x {"a" "value"}] (print x))', 'output': '{a value}'},
            {'src': '(let [x-y {}] (print x-y))', 'output': '{}'},
            {'src': '(let [x-y {}] (let [z-a 1] (print x-y)))', 'output': '{}'},
            {'src': '(let [x-y {}] (print x-y) (print "done"))', 'output': '{}done'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_loops(self):
        tests = [
            {'src': '(print (loop [cnt 3 acc 0] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc 1)))))', 'output': '3'},
            {'src': '(print (loop [cnt 3 acc 0] (if (= 0 cnt) "3" (recur (- cnt 1) (+ acc 1)))))', 'output': '3'},
            {'src': '(print (loop [cnt 3 acc 0] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc cnt)))))', 'output': '6'},
            {'src': '(loop [n 0] (do (print n) (if (< n 2) (recur (+ n 1)))))', 'output': '012'},
            {'src': '(loop [n 0] (print n) (if (< n 2) (recur (+ n 1))))', 'output': '012'},
            {'src': '(loop [n 0] (print (str n)) (if (< n 2) (recur (+ n 1))))', 'output': '012'},
            {'src': '(loop [n 0] (if (> n 2) (print n) (let [y 1] (recur (+ n y)))))', 'output': '3'},
            {'src': '(let [b 2] (loop [n 0] (if (> n b) (print n) (let [y 1] (recur (+ n y))))))', 'output': '3'},
            {'src': '(let [b 2] (loop [n 0] (if (> n b) (print (str n)) (let [y 1] (recur (+ n y))))))', 'output': '3'},
            {'src': '(loop [n 0] (if (> n 2) (print "done") (do (print (str n)) (recur (+ n 1)))))', 'output': '012done'},
            {'src': '(loop [n 0] (do (print n) (print "    ") (println (/ (* 5 (- n 32)) 9)) (if (< n 70) (recur (+ 20 n)))))', 'output': f'0    -17.7778{LSEP}20    -6.66667{LSEP}40    4.44444{LSEP}60    15.5556{LSEP}80    26.6667{LSEP}'},
            {'src': '(print (do (loop [cnt 3 acc 0] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc 1)))) (loop [cnt 3 acc 0] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc 1))))))', 'output': '3'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_fn(self):
        tests = [
            {'src': '(print ((fn [x] x) 1))', 'output': '1'},
            {'src': '(print ((fn [x] x) "string"))', 'output': 'string'},
            {'src': '((fn [] (print "function")))', 'output': 'function'},
            {'src': '((fn [x] (print x)) "string")', 'output': 'string'},
            {'src': '(print ((fn [x y] (+ x y)) 1 2))', 'output': '3'},
            {'src': '((fn [x] (print x) (print "done")) 1)', 'output': '1done'},
            {'src': '(print ((fn [cnt acc] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc 1)))) 3 0))', 'output': '3'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_def(self):
        tests = [
            {'src': '(def a 1) (print a)', 'output': '1'},
            {'src': '(def a "1") (print a)', 'output': '1'},
            {'src': '(def some-thing 1) (print some-thing)', 'output': '1'},
            {'src': '(def some-thing "1") (print some-thing)', 'output': '1'},
            {'src': '(def some-thing {"a" "b"}) (print some-thing)', 'output': '{a b}'},
            {'src': '(def a "1") (def a "2") (print a)', 'output': '2'},
            {'src': '(def some-thing (assoc {} "a" "value")) (print some-thing)', 'output': '{a value}'},
            {'src': '(def some-thing (let [d {"a" "value"}] (assoc d "b" "another"))) (print some-thing)', 'output': '{a value, b another}'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_defn(self):
        tests = [
            {'src': '(defn f1 [x y] (+ x y)) (print (f1 1 2))', 'output': '3'},
            {'src': '(defn f1 [x y] (print "function")) (f1 1 2)', 'output': 'function'},
            {'src': '(defn f1 [x y] (print "function") (print "done") "return") (print (f1 1 2))', 'output': 'functiondonereturn'},
            {'src': '(defn f-1 [x y] (+ x y)) (print (f-1 1 2))', 'output': '3'},
            {'src': '(defn f-1 [cnt acc] (if (= 0 cnt) (str acc) (recur (- cnt 1) (+ acc 1)))) (print (f-1 3 0))', 'output': '3'},
            {'src': '(defn f-1 [a] (if (> a 2) a (f-1 (+ a 1)))) (print (f-1 0))', 'output': '3'},
            {'src': '(defn fib [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))) (println (fib 5))', 'output': f'5{LSEP}'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_for(self):
        tests = [
            {'src': '(for [w ["one" "two"]] (print w))', 'output': 'onetwo'},
            {'src': '(print (for [w ["one" "two"]] (print w)))', 'output': 'onetwonil'},
            {'src': '(let [a "1"] (for [w ["one" "two"]] (print w)))', 'output': 'onetwo'},
            {'src': '(let [a "1"] (for [w ["one" "two"]] (print [w a])))', 'output': '[one 1][two 1]'},
            {'src': '(for [w ["one" "two"]] (print "item ") (println w))', 'output': f'item one{LSEP}item two{LSEP}'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_string_module(self):
        tests = [
            {'src': '(require [language.string str]) (print (str/blank? "Hello World"))', 'output': 'false'},
            {'src': '(require [language.string str]) (print (str/blank? ""))', 'output': 'true'},
            {'src': '(require [language.string str]) (print (str/blank? nil))', 'output': 'true'},
            {'src': '(require [language.string str]) (print (str/blank? "\\n"))', 'output': 'true'},
            {'src': '(require [language.string str]) (print (str/lower "Hello"))', 'output': 'hello'},
            {'src': '(require [language.string str]) (print (str/lower "Hello World"))', 'output': 'hello world'},
            {'src': '(require [language.string str]) (print (str/split "hello"))', 'output': '[hello]'},
            {'src': '(require [language.string str]) (print (str/split "ab cd"))', 'output': '[ab cd]'},
            {'src': '(require [language.string str]) (print (str/split "hello world"))', 'output': '[hello world]'},
            {'src': '(require [language.string str]) (print (str/split "hellohello worldworld"))', 'output': '[hellohello worldworld]'},
            {'src': '(require [language.string str]) (print (str/index-of "abcd" 1))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/index-of "abcdefgh" 1))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/index-of 1 "a"))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/index-of 1 "abcdefgh"))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/index-of "abcd" "j"))', 'output': 'nil'},
            {'src': '(require [language.string str]) (print (str/index-of "abcd" "de"))', 'output': 'nil'},
            {'src': '(require [language.string str]) (print (str/index-of "abcd" "abcde"))', 'output': 'nil'},
            {'src': '(require [language.string str]) (print (str/index-of "abcd" "abcdefgh"))', 'output': 'nil'},
            {'src': '(require [language.string str]) (print (str/index-of "abcd" "ab"))', 'output': '0'},
            {'src': '(require [language.string str]) (print (str/index-of "abcd" "bc"))', 'output': '1'},
            {'src': '(require [language.string str]) (print (str/index-of "abcd" "abcd"))', 'output': '0'},
            {'src': '(require [language.string str]) (print (str/index-of "abcdefgh" "j"))', 'output': 'nil'},
            {'src': '(require [language.string str]) (print (str/index-of "abcdefgh" "a"))', 'output': '0'},
            {'src': '(require [language.string str]) (print (str/index-of "abcdefgh abcd" "abcdefgh"))', 'output': '0'},
            {'src': '(require [language.string str]) (print (str/index-of "abcdefgh abcd" "cdefgh a"))', 'output': '2'},
            {'src': '(require [language.string str]) (print (str/index-of "abcdefgh" "abcdefgh"))', 'output': '0'},
            {'src': '(require [language.string str]) (print (str/index-of "aabaa" "aa" 2))', 'output': '3'},
            {'src': '(require [language.string str]) (print (str/index-of "aabcdefgh aa" "aa" 2))', 'output': '10'},
            {'src': '(require [language.string str]) (print (str/subs 1 1))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/subs "abcd" "a"))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/subs "abcd" 1 "a"))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" "a"))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" 1 "a"))', 'output': 'ERROR: Type'},
            {'src': '(require [language.string str]) (print (str/subs "abcd" 2))', 'output': 'cd'},
            {'src': '(require [language.string str]) (print (str/subs "abcd" 2 3))', 'output': 'c'},
            {'src': '(require [language.string str]) (print (str/subs "abcd" 2 2))', 'output': ''},
            {'src': '(require [language.string str]) (print (str/subs "abcd" 2 1))', 'output': 'ERROR: OutOfBounds'},
            {'src': '(require [language.string str]) (print (str/subs "abcd" 2 8))', 'output': 'ERROR: OutOfBounds'},
            {'src': '(require [language.string str]) (print (str/subs "abcd" 2 -1))', 'output': 'ERROR: OutOfBounds'},
            {'src': '(require [language.string str]) (print (str/subs "abcd" -2 -1))', 'output': 'ERROR: OutOfBounds'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" 2))', 'output': 'cdefgh'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" 0))', 'output': 'abcdefgh'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" 2 3))', 'output': 'c'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" 2 2))', 'output': ''},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" 2 1))', 'output': 'ERROR: OutOfBounds'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" 2 15))', 'output': 'ERROR: OutOfBounds'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" 2 -1))', 'output': 'ERROR: OutOfBounds'},
            {'src': '(require [language.string str]) (print (str/subs "abcdefgh" -2 -1))', 'output': 'ERROR: OutOfBounds'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_advanced(self):
        tests = [
            {'src': '(print (str []))', 'output': '[]'},
            {'src': '(print (str {}))', 'output': '{}'},
            {'src': '(require [language.string str]) (print (if (str/blank? "Hello World") "blank" "not blank"))', 'output': 'not blank'},
            {'src': '(print (if "Hello World" (read-line) "false"))', 'input': 'line', 'output': 'line'},
            {'src': '(def a 1) (let [b 2] (print (+ a b)))', 'output': '3'},
            {'src': '(print (let [b {"key" (get {"z" 2} "z")}] (+ 1 (get b "key"))))', 'output': '3'},
            {'src': '(let [x 1] (if (= x 1) (print true) (print false)))', 'output': 'true'},
            {'src': '(let [x-y 1] (do (print x-y) (print "done")))', 'output': '1done'},
            {'src': '(let [x-y 1] (loop [n-p 0] (if (= n-p 1) (print x-y) (recur (+ n-p 1)))))', 'output': '1'},
            {'src': '(let [a 1] (let [b 2] (print (+ a b))))', 'output': '3'},
            {'src': '(def d {}) (assoc d "a" 1) (print (get d "a"))', 'output': '1'},
            {'src': '(def i 0) (print ((fn [n] n) i))', 'output': '0'},
            {'src': '(print ((fn [n] (loop [cnt n acc 1] (if (= 0 cnt) acc (recur (- cnt 1) (* acc cnt))))) 3))', 'output': '6'},
            {'src': '(defn f1 [x] (let [y (+ x 1)] y)) (print (f1 1))', 'output': '2'},
            {'src': '(defn f1 [x] (+ x 1)) (let [y 1] (print (f1 y)))', 'output': '2'},
            {'src': '(defn f1 [z] (let [x 1] (loop [y 2] (if (< y 5) (recur (+ y 1)) (+ x z))))) (print (f1 3))', 'output': '4'},
            {'src': '(defn f1 [] {"key" "value"}) (print (get (f1) "key"))', 'output': 'value'},
            {'src': '(defn compare [a b] (> (nth a 1) (nth b 1))) (print (sort compare [["a" 1] ["b" 2]]))', 'output': '[[b 2] [a 1]]'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) "done" (recur (read-line)))))', 'input': '', 'output': 'done'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) "done" (recur (read-line)))))', 'input': 'line', 'output': 'done'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) ["done"] (recur (read-line)))))', 'input': 'line', 'output': '[done]'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) {"a" "b"} (recur (read-line)))))', 'input': 'line', 'output': '{a b}'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) "done" (do (print line) (recur (read-line))))))', 'input': f'line1{LSEP}line2{LSEP}', 'output': 'line1line2done'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_other(self):
        tests = [
            {'src': '(println "hello")', 'output': f'hello{LSEP}'},
            {'src': '(print (read-line))', 'input': f'line{LSEP}', 'output': 'line'},
            {'src': '(print (read-line))', 'input': LSEP, 'output': ''},
            {'src': '(print (read-line))', 'input': '', 'output': 'nil'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_file_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_name = os.path.join(tmp, 'file')
            with open(file_name, 'wb') as f:
                f.write('asdf'.encode('utf8'))
            file_name = file_name.replace('\\', '\\\\')
            test = {'src': f'(print (let [f (file/open "{file_name}"), data (file/read f)] (do (file/close f) data)))', 'input': '', 'output': 'asdf'}
            _run_test(test, self.assertEqual)

    def test_file_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_name = os.path.join(tmp, 'file')
            file_name = file_name.replace('\\', '\\\\')
            test = {'src': f'(let [f (file/open "{file_name}" "w")] (file/write f "asdf") (file/close f)) (print (let [f (file/open "{file_name}") data (file/read f)] data))', 'input': '', 'output': 'asdf'}
            _run_test(test, self.assertEqual)

    def test_mkdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            dir_name = os.path.join(tmp, 'testdir')
            dir_name = dir_name.replace('\\', '\\\\')
            test = {'src': f'(require [language.os os]) (os/mkdir "{dir_name}")', 'input': '', 'output': ''}
            _run_test(test, self.assertEqual)
            self.assertTrue(os.path.exists(dir_name))

    def test_math_gcd(self):
        tests = [
            {'src': '(require [language.math math]) (print (math/gcd "a" 9))', 'output': 'ERROR: Type'},
            {'src': '(require [language.math math]) (print (math/gcd 1 "a"))', 'output': 'ERROR: Type'},
            {'src': '(require [language.math math]) (print (math/gcd 1.5 2))', 'output': 'ERROR: Type'},
            {'src': '(require [language.math math]) (print (math/gcd 0 0))', 'output': '0'},
            {'src': '(require [language.math math]) (print (math/gcd 0 1))', 'output': '1'},
            {'src': '(require [language.math math]) (print (math/gcd 5 0))', 'output': '5'},
            {'src': '(require [language.math math]) (print (math/gcd 1 1))', 'output': '1'},
            {'src': '(require [language.math math]) (print (math/gcd 2 1))', 'output': '1'},
            {'src': '(require [language.math math]) (print (math/gcd 2 2))', 'output': '2'},
            {'src': '(require [language.math math]) (print (math/gcd 2 3))', 'output': '1'},
            {'src': '(require [language.math math]) (print (math/gcd 3 6))', 'output': '3'},
            {'src': '(require [language.math math]) (print (math/gcd -3 6))', 'output': '3'},
            {'src': '(require [language.math math]) (print (math/gcd 3 9))', 'output': '3'},
            {'src': '(require [language.math math]) (print (math/gcd 15 50))', 'output': '5'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_sqlite(self):
        # _build_sqlite([clang_cmd])
        CREATE = 'CREATE TABLE data (id INTEGER PRIMARY KEY, col1 TEXT)'
        INSERT = 'INSERT INTO data (id, col1) VALUES (1, \'something\');'
        SELECT = 'SELECT * FROM data;'
        tests = [
            {'src': '(require [language.sqlite3 s]) (print (s/version))', 'output': f'3.41.2'},
            {'src': '(require [language.sqlite3 s]) (let [db (s/open ":memory:")] (print (s/close db))', 'output': f'nil'},
            {'src': f'(require [language.sqlite3 sqlite3]) (let [db (sqlite3/open ":memory:")] (sqlite3/execute db "{CREATE}") (sqlite3/execute db "{INSERT}") (print (sqlite3/execute db "{SELECT}")) (sqlite3/close db))', 'output': f'[[1 something]]'},
            {'src': f'(require [language.sqlite3 sqlite3]) (with [db (sqlite3/open ":memory:")] (sqlite3/execute db "{CREATE}") (sqlite3/execute db "{INSERT}") (print (sqlite3/execute db "{SELECT}")))', 'output': f'[[1 something]]'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual, sqlite=True)


class ProgramTests(unittest.TestCase):
    def test_run_program(self):
        program_file = 'test_program.clj'
        cmd = f'{sys.executable} lang.py -r {program_file} arg1 arg2'
        input_ = f'one{LSEP}two{LSEP}one'
        try:
            result = subprocess.run(cmd.split(), check=True, input=input_.encode('utf8'), capture_output=True)
            result_output = result.stdout.decode('utf8')
            self.assertEqual(result_output, '{one 2, two 1}a%sarg1arg2%s3.41.2' % (LSEP, LSEP))
        except subprocess.CalledProcessError as e:
            print(e.stderr.decode("utf8"))
            raise RuntimeError(f'run_program test failed: {e}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-q':
            QUICK = True
            sys.argv.remove('-q')
        elif sys.argv[1] == '--ci':
            CI = True
            sys.argv.remove('--ci')

    unittest.main()
