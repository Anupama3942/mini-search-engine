"""
Mini Search Engine - Query Parser & AST Evaluator
Stage 11: Optimized Evaluation with Early Termination & Candidate Pruning
"""

import re
from fuzzy_search import resolve_term

class ASTNode:
    pass

class TermNode(ASTNode):
    def __init__(self, term):
        self.term = term
    def __repr__(self):
        return f"TERM({self.term})"

class PhraseNode(ASTNode):
    def __init__(self, terms):
        self.terms = terms
    def __repr__(self):
        return f"PHRASE({self.terms})"

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
    Tokenize the query string into Boolean operators, parentheses, phrases, and processed terms.
    """
    if query.count('"') % 2 != 0:
        raise ValueError("Invalid phrase query. Please close the quotation marks.")

    tokens = []
    pattern = r'("[^"]*")|(\()|(\))|([^\s()]+)'
    
    for match in re.finditer(pattern, query):
        quoted, open_p, close_p, word = match.groups()
        
        if quoted:
            if quoted == '""':
                raise ValueError("Empty phrases are not supported.")
            phrase_content = quoted[1:-1].strip()
            if not phrase_content:
                raise ValueError("Empty phrases are not supported.")
                
            phrase_terms = process_text_func(phrase_content)
            if not phrase_terms:
                raise ValueError("Empty phrases are not supported.")
                
            tokens.append(("PHRASE", phrase_terms))
            
        elif open_p:
            tokens.append('(')
        elif close_p:
            tokens.append(')')
        elif word:
            if word.upper() in ('AND', 'OR', 'NOT'):
                tokens.append(word.upper())
            else:
                processed = process_text_func(word)
                for pt in processed:
                    tokens.append(pt)
                    
    return tokens

class QueryParser:
    """Recursive descent parser for Boolean and Phrase queries."""
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
        while True:
            if self.current() == 'AND':
                self.consume()
                right = self.parse_not()
                node = AndNode(node, right)
            elif self.current() not in ('OR', ')', None):
                # Implicit AND for adjacent terms, phrases, or parenthesized expressions
                right = self.parse_not()
                node = AndNode(node, right)
            else:
                break
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
            
        if isinstance(token, tuple) and token[0] == 'PHRASE':
            return PhraseNode(token[1])
            
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

def resolve_ast(node, vocabulary, cache=None):
    """
    Traverse the AST and resolve unknown TermNodes using fuzzy matching.
    Phrase contents remain exact.
    """
    if isinstance(node, TermNode):
        resolved, is_fuzzy, _ = resolve_term(node.term, vocabulary, cache)
        if is_fuzzy:
            return TermNode(resolved), {node.term: resolved}
        return node, {}
        
    elif isinstance(node, PhraseNode):
        return node, {}
        
    elif isinstance(node, AndNode):
        left_node, left_corr = resolve_ast(node.left, vocabulary, cache)
        right_node, right_corr = resolve_ast(node.right, vocabulary, cache)
        return AndNode(left_node, right_node), {**left_corr, **right_corr}
        
    elif isinstance(node, OrNode):
        left_node, left_corr = resolve_ast(node.left, vocabulary, cache)
        right_node, right_corr = resolve_ast(node.right, vocabulary, cache)
        return OrNode(left_node, right_node), {**left_corr, **right_corr}
        
    elif isinstance(node, NotNode):
        child_node, child_corr = resolve_ast(node.child, vocabulary, cache)
        return NotNode(child_node), child_corr
        
    return node, {}

def evaluate_phrase(phrase_terms, positional_index):
    """
    Optimized phrase evaluation:
      1. Intersect document candidate sets first (smallest set first).
      2. Early exit if intersection is empty.
      3. Positional adjacency check only on qualified candidates.
    """
    if not phrase_terms:
        return set()
        
    # Check if all terms exist in index
    for t in phrase_terms:
        if t not in positional_index:
            return set()

    first_term = phrase_terms[0]
    candidate_docs = set(positional_index[first_term].keys())
    
    # Intersect with all other terms
    for term in phrase_terms[1:]:
        if not candidate_docs:
            return set() # Early exit
        candidate_docs &= set(positional_index[term].keys())
        
    if not candidate_docs:
        return set()

    matching_docs = set()
    for doc in candidate_docs:
        valid_doc = False
        first_term_positions = positional_index[first_term][doc]
        
        for pos in first_term_positions:
            valid_sequence = True
            for i, term in enumerate(phrase_terms[1:], start=1):
                if (pos + i) not in positional_index[term][doc]:
                    valid_sequence = False
                    break
                    
            if valid_sequence:
                valid_doc = True
                break
                
        if valid_doc:
            matching_docs.add(doc)
            
    return matching_docs

def evaluate_query(node, inverted_index, all_documents, positional_index):
    """
    Recursively evaluate the AST with early-termination on empty intersections.
    """
    if isinstance(node, TermNode):
        return set(inverted_index.get(node.term, set()))
        
    elif isinstance(node, PhraseNode):
        return evaluate_phrase(node.terms, positional_index)
    
    elif isinstance(node, AndNode):
        left_result = evaluate_query(node.left, inverted_index, all_documents, positional_index)
        # Early termination optimization
        if not left_result:
            return set()
        right_result = evaluate_query(node.right, inverted_index, all_documents, positional_index)
        return left_result & right_result
        
    elif isinstance(node, OrNode):
        left_result = evaluate_query(node.left, inverted_index, all_documents, positional_index)
        right_result = evaluate_query(node.right, inverted_index, all_documents, positional_index)
        return left_result | right_result
        
    elif isinstance(node, NotNode):
        child_result = evaluate_query(node.child, inverted_index, all_documents, positional_index)
        return all_documents - child_result

    return set()

def extract_positive_terms(node):
    """
    Extracts all query terms that are NOT part of a NOT expression.
    """
    if isinstance(node, TermNode):
        return [node.term]
    elif isinstance(node, PhraseNode):
        return list(node.terms)
    elif isinstance(node, AndNode):
        return extract_positive_terms(node.left) + extract_positive_terms(node.right)
    elif isinstance(node, OrNode):
        return extract_positive_terms(node.left) + extract_positive_terms(node.right)
    elif isinstance(node, NotNode):
        return []
    return []
