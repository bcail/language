import unittest
from lang import TokenType, scan_tokens, parse


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


if __name__ == '__main__':
    unittest.main()
