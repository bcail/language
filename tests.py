import unittest
from lang import TokenType, scan_tokens, parse, evaluate


SOURCE = '(+ 10 2 (- 15 (+ 4 4)) 5)'
EXPECTED_TOKENS = [
    {'lexeme': '(', 'type': TokenType.LEFT_PAREN},
    {'lexeme': '+', 'type': TokenType.PLUS},
    {'lexeme': '10', 'type': TokenType.NUMBER},
    {'lexeme': '2', 'type': TokenType.NUMBER},
    {'lexeme': '(', 'type': TokenType.LEFT_PAREN},
    {'lexeme': '-', 'type': TokenType.MINUS},
    {'lexeme': '15', 'type': TokenType.NUMBER},
    {'lexeme': '(', 'type': TokenType.LEFT_PAREN},
    {'lexeme': '+', 'type': TokenType.PLUS},
    {'lexeme': '4', 'type': TokenType.NUMBER},
    {'lexeme': '4', 'type': TokenType.NUMBER},
    {'lexeme': ')', 'type': TokenType.RIGHT_PAREN},
    {'lexeme': ')', 'type': TokenType.RIGHT_PAREN},
    {'lexeme': '5', 'type': TokenType.NUMBER},
    {'lexeme': ')', 'type': TokenType.RIGHT_PAREN},
]

EXPECTED_AST = [
    {'lexeme': '+', 'type': TokenType.PLUS},
    10,
    2,
    [
        {'lexeme': '-', 'type': TokenType.MINUS},
        15,
        [
            {'lexeme': '+', 'type': TokenType.PLUS},
            4,
            4,
        ]
    ],
    5,
]


class ScanTokenTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        tokens = scan_tokens(SOURCE)
        self.assertEqual(tokens, EXPECTED_TOKENS)


class ParseTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        ast = parse(EXPECTED_TOKENS)
        self.assertEqual(ast, EXPECTED_AST)


class EvalTests(unittest.TestCase):

    def test(self):
        tests = [
            {'src': '(+ 1 2)', 'result': 3},
            {'src': '(+ 1 (+ 1 1))', 'result': 3},
            {'src': '(+ 1 (- 4 2))', 'result': 3},
            {'src': '(* 2 (- 4 2) 3)', 'result': 12},
            {'src': '(/ 12 (- 4 2) 3)', 'result': 2},
            {'src': '(+ 10 -2)', 'result': 8},
        ]

        for test in tests:
            with self.subTest(test=test):
                self.assertEqual(evaluate(parse(scan_tokens(test['src']))), test['result'])


if __name__ == '__main__':
    unittest.main()
