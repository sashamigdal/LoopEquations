"""Minimal Mathematica .nb parser: extracts cells and converts boxes to linear text."""
import re, sys

TOK = re.compile(r'''
 (?P<ws>\s+)
|(?P<comment>\(\*.*?\*\))
|(?P<str>"(?:[^"\\]|\\.)*")
|(?P<num>-?\d+\.?\d*(?:`[\d.]*)?(?:\*\^-?\d+)?)
|(?P<rule>:>|->)
|(?P<sym>[A-Za-z$][A-Za-z0-9$`]*)
|(?P<lb>\[)|(?P<rb>\])|(?P<lc>\{)|(?P<rc>\})|(?P<comma>,)
|(?P<other>.)
''', re.S | re.X)


class Expr:
    __slots__ = ('head', 'args')
    def __init__(self, head, args):
        self.head = head; self.args = args
    def __repr__(self):
        return f'{self.head}[{", ".join(map(repr, self.args))}]'


class Rule:
    __slots__ = ('lhs', 'rhs')
    def __init__(self, l, r):
        self.lhs = l; self.rhs = r


def tokenize(s):
    for m in TOK.finditer(s):
        k = m.lastgroup
        if k in ('ws', 'comment'):
            continue
        yield k, m.group()


def parse(s):
    toks = list(tokenize(s))
    pos = 0

    def atom():
        nonlocal pos
        k, v = toks[pos]
        if k == 'lc':
            pos += 1
            items = seq('rc')
            return Expr('List', items)
        if k == 'str':
            pos += 1
            return ('S', unescape(v[1:-1]))
        if k == 'num':
            pos += 1
            return ('N', v)
        if k == 'sym':
            pos += 1
            e = ('Y', v)
            while pos < len(toks) and toks[pos][0] == 'lb':
                pos += 1
                args = seq('rb')
                e = Expr(e[1] if isinstance(e, tuple) else e, args)
            return e
        if k == 'lb':
            pos += 1
            return Expr('Bracket', seq('rb'))
        pos += 1
        return ('O', v)

    def item():
        nonlocal pos
        a = atom()
        if pos < len(toks) and toks[pos][0] == 'rule':
            pos += 1
            b = item()
            return Rule(a, b)
        return a

    def seq(close):
        nonlocal pos
        items = []
        if pos >= len(toks):
            return items
        if toks[pos][0] == close:
            pos += 1
            return items
        while True:
            if pos >= len(toks):
                return items
            if toks[pos][0] in ('rb','rc') and toks[pos][0] != close:
                pos += 1
                continue
            items.append(item())
            if pos >= len(toks):
                return items
            k, v = toks[pos]
            if k == 'comma':
                pos += 1
                continue
            if k == close:
                pos += 1
                return items
            # skip garbage (operators etc.)
            if k in ('rb','rc'):
                pos += 1
            

    return item()


NAMED = {
    'IndentingNewLine': '\n', 'NewLine': '\n', 'Equal': '==', 'Rule': '->', 'RuleDelayed': ':>',
    'LessEqual': '<=', 'GreaterEqual': '>=', 'NotEqual': '!=', 'Infinity': 'Infinity',
    'Pi': 'Pi', 'ImaginaryI': 'I', 'ExponentialE': 'E', 'Element': '∈', 'InvisibleSpace': '',
    'InvisibleTimes': '', 'Times': '*', 'Cross': 'x', 'CenterDot': '.', 'Rule ': '->',
    'Function': '|->', 'DifferentialD': 'd', 'PartialD': 'D', 'Integral': 'Integrate',
    'LeftDoubleBracket': '[[', 'RightDoubleBracket': ']]', 'And': '&&', 'Or': '||', 'Not': '!',
    'Sum': 'Sum', 'Product': 'Product', 'Transpose': '^T', 'Conjugate': '*', 'Degree': 'Degree',
    'Minus': '-', 'Divide': '/', 'LongRightArrow': '-->', 'RightArrow': '->', 'Implies': '=>',
    'SmallCircle': '@', 'Distributed': '~', 'Colon': ':', 'VeryThinSpace': ' ', 'ThinSpace': ' ',
    'MediumSpace': ' ', 'ThickSpace': ' ', 'NegativeVeryThinSpace': '', 'Hyphen': '-',
    'Dash': '-', 'LongDash': '--', 'Bullet': '*', 'Ellipsis': '...', 'Placeholder': '_',
    'SelectionPlaceholder': '_', 'LeftAngleBracket': '<', 'RightAngleBracket': '>',
    'Sqrt': 'Sqrt', 'Mu': 'mu', 'Nu': 'nu', 'Alpha': 'alpha', 'Beta': 'beta', 'Gamma': 'gamma',
    'Delta': 'delta', 'Epsilon': 'epsilon', 'Lambda': 'lambda', 'Sigma': 'sigma', 'Tau': 'tau',
    'Theta': 'theta', 'Phi': 'phi', 'Psi': 'psi', 'Omega': 'omega', 'Xi': 'xi', 'Eta': 'eta',
    'Zeta': 'zeta', 'Kappa': 'kappa', 'Rho': 'rho', 'Chi': 'chi', 'CurlyEpsilon': 'epsilon',
    'CapitalGamma': 'Gamma', 'CapitalDelta': 'Delta', 'CapitalLambda': 'Lambda',
    'CapitalSigma': 'Sigma', 'CapitalOmega': 'Omega', 'CapitalPhi': 'Phi', 'CapitalPsi': 'Psi',
    'CapitalTheta': 'Theta', 'CapitalXi': 'Xi', 'CapitalPi': 'PPi',
    'Iota': 'iota', 'Omicron': 'omicron', 'Upsilon': 'upsilon', 'CurlyPhi': 'phi', 'CurlyTheta': 'theta',
    'ScriptCapitalL': 'ScriptL', 'ScriptCapitalN': 'ScriptN', 'ScriptCapitalM': 'ScriptM',
    'CapitalEpsilon': 'Epsilon', 'Star': '*', 'Equilibrium': '<=>', 'Tilde': '~', 'Prime': "'",
    'CirclePlus': '(+)', 'CircleTimes': '(x)', 'PlusMinus': '+-', 'TildeTilde': '~~',
    'Congruent': '===', 'Proportional': '~', 'Approx': '~', 'TildeEqual': '~=',
    'DirectedEdge': '->', 'UndirectedEdge': '<->', 'Sum ': 'Sum', 'Cap': 'cap', 'Cup': 'cup',
    'Subset': 'subset', 'Superset': 'superset', 'ForAll': 'forall', 'Exists': 'exists',
    'Rightarrow': '=>', 'LeftRightArrow': '<->', 'Hbar': 'hbar', 'Micro': 'mu', 'Angstrom': 'A',
    'SpanFromLeft': '', 'SpanFromAbove': '', 'AlignmentMarker': '', 'Null': '',
}


def unescape(s):
    s = s.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
    s = re.sub(r'\\\[(\w+)\]', lambda m: NAMED.get(m.group(1), '\\[' + m.group(1) + ']'), s)
    s = s.replace('\\<', '').replace('\\>', '').replace('\\\n', '').replace('\\\\', '\\')
    return s


def txt(e):
    if isinstance(e, tuple):
        return e[1]
    if isinstance(e, Rule):
        return txt(e.lhs) + '->' + txt(e.rhs)
    h = e.head
    a = e.args
    if isinstance(h, Expr):
        return repr(e)
    if h == 'List':
        return ''.join(txt(x) for x in a)
    if h == 'RowBox':
        return txt(a[0])
    if h == 'BoxData':
        return txt(a[0])
    if h == 'SuperscriptBox':
        return f'{par(a[0])}^{par(a[1])}'
    if h == 'SubscriptBox':
        return f'{txt(a[0])}_{txt(a[1])}'
    if h == 'SubsuperscriptBox':
        return f'{txt(a[0])}_{txt(a[1])}^{par(a[2])}'
    if h == 'FractionBox':
        return f'({txt(a[0])})/({txt(a[1])})'
    if h == 'SqrtBox':
        return f'Sqrt[{txt(a[0])}]'
    if h == 'RadicalBox':
        return f'Surd[{txt(a[0])},{txt(a[1])}]'
    if h in ('StyleBox', 'TagBox', 'InterpretationBox', 'FormBox', 'TooltipBox', 'ButtonBox',
             'AdjustmentBox', 'FrameBox', 'PaneBox', 'ItemBox', 'StyleData', 'DynamicBox'):
        return txt(a[0]) if a else ''
    if h == 'UnderoverscriptBox':
        return f'{txt(a[0])}_({txt(a[1])})^({txt(a[2])})'
    if h in ('UnderscriptBox', 'OverscriptBox'):
        return f'{txt(a[0])}[{txt(a[1])}]'
    if h == 'TemplateBox':
        return 'TemplateBox[' + ','.join(txt(x) for x in (a[0].args if isinstance(a[0], Expr) else [a[0]])) + ']'
    if h == 'GridBox':
        rows = a[0]
        out = []
        for r in rows.args:
            out.append(' | '.join(txt(c) for c in (r.args if isinstance(r, Expr) else [r])))
        return '\n' + '\n'.join(out) + '\n'
    if h == 'Cell':
        return txt(a[0])
    if h == 'TextData':
        return txt(a[0])
    if h in ('GraphicsBox', 'Graphics3DBox', 'RasterBox', 'DynamicModuleBox'):
        return f'<{h}>'
    if h == 'Rule':
        return ''
    return f'{h}[...]'


def par(e):
    s = txt(e)
    if re.fullmatch(r'[\w.$`]+', s):
        return s
    return '(' + s + ')'


def cells(e, out):
    if isinstance(e, Expr):
        if e.head == 'Cell' and len(e.args) >= 2 and isinstance(e.args[1], tuple) and e.args[1][0] == 'S':
            style = e.args[1][1]
            label = ''
            for r in e.args[2:]:
                if isinstance(r, Rule) and r.lhs == ('Y', 'CellLabel'):
                    label = txt(r.rhs)
            out.append((style, label, e.args[0]))
            return
        for x in e.args:
            cells(x, out)
    elif isinstance(e, Rule):
        cells(e.rhs, out)


if __name__ == '__main__':
    src = open(sys.argv[1], encoding='latin-1').read()
    i = src.index('Notebook[')
    j = src.find('(* End of Notebook Content *)')
    tree = parse(src[i:j])
    out = []
    cells(tree, out)
    maxout = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    for n, (style, label, content) in enumerate(out):
        t = txt(content)
        if style in ('Output', 'Print', 'Message') and len(t) > maxout:
            t = t[:maxout] + f'... [truncated {len(t)} chars]'
        print(f'==== [{n}] {style} {label}')
        print(t)
