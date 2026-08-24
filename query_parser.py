class ASTNode:
    pass

class TermNode(ASTNode):
    def __init__(self, term):
        self.term = term
    def __repr__(self):
        return f"TERM({self.term})"

class AndNode(ASTNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"AND({self.left}, {self.right})"

class OrNode(ASTNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"OR({self.left}, {self.right})"

class NotNode(ASTNode):
    def __init__(self, child):
        self.child = child
    def __repr__(self):
        return f"NOT({self.child})"

def tokenize_query(query, process_text_func):
    """
    Tokenize the query string into Boolean operators, parentheses, and processed terms.
    """
    if '"' in query:
        raise ValueError("Phrase search is not available yet.")

    # Pad parentheses to make splitting easy
    q = query.replace('(', ' ( ').replace(')', ' ) ')
    raw_tokens = q.split()

    tokens = []
    for t in raw_tokens:
        if t == '(':
            tokens.append('(')
        elif t == ')':
            tokens.append(')')
        elif t.upper() in ('AND', 'OR', 'NOT'):
            tokens.append(t.upper())
        else:
            # Process the normal terms using the Stage 3 text pipeline
            processed = process_text_func(t)
            for pt in processed:
                tokens.append(pt)
                
    return tokens

class QueryParser:
    """Recursive descent parser for Boolean queries."""
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        token = self.current()
        self.pos += 1
        return token

    def parse(self):
        if not self.tokens:
            return None
        node = self.parse_or()
        if self.current() is not None:
            raise ValueError("Invalid search query. Please check your Boolean operators and parentheses.")
        return node

    def parse_or(self):
        node = self.parse_and()
        while self.current() == 'OR':
            self.consume()
            right = self.parse_and()
            node = OrNode(node, right)
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.current() == 'AND':
            self.consume()
            right = self.parse_not()
            node = AndNode(node, right)
        return node

    def parse_not(self):
        if self.current() == 'NOT':
            self.consume()
            child = self.parse_not()
            return NotNode(child)
        return self.parse_primary()

    def parse_primary(self):
        token = self.consume()
        if token is None:
            raise ValueError("Invalid search query. Missing term or parenthesis.")
        if token == '(':
            node = self.parse_or()
            if self.consume() != ')':
                raise ValueError("Invalid search query. Missing closing parenthesis.")
            return node
        elif token == ')':
            raise ValueError("Invalid search query. Unexpected closing parenthesis.")
        elif token in ('AND', 'OR', 'NOT'):
            raise ValueError(f"Invalid search query. Unexpected operator '{token}'.")
        else:
            return TermNode(token)

def evaluate_query(node, inverted_index, all_documents):
    """
    Recursively evaluates the AST and returns a set of matching documents.
    """
    if isinstance(node, TermNode):
        return set(inverted_index.get(node.term, set()))
    
    elif isinstance(node, AndNode):
        left_result = evaluate_query(node.left, inverted_index, all_documents)
        right_result = evaluate_query(node.right, inverted_index, all_documents)
        return left_result & right_result
        
    elif isinstance(node, OrNode):
        left_result = evaluate_query(node.left, inverted_index, all_documents)
        right_result = evaluate_query(node.right, inverted_index, all_documents)
        return left_result | right_result
        
    elif isinstance(node, NotNode):
        child_result = evaluate_query(node.child, inverted_index, all_documents)
        return all_documents - child_result

    return set()

def extract_positive_terms(node):
    """
    Extracts all query terms that are NOT part of a NOT expression.
    These are the terms that should contribute to the TF-IDF score.
    """
    if isinstance(node, TermNode):
        return [node.term]
    elif isinstance(node, AndNode):
        return extract_positive_terms(node.left) + extract_positive_terms(node.right)
    elif isinstance(node, OrNode):
        return extract_positive_terms(node.left) + extract_positive_terms(node.right)
    elif isinstance(node, NotNode):
        # We deliberately ignore terms under a NOT node for positive scoring!
        return []
    return []
