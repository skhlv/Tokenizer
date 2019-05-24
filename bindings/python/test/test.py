# -*- coding: utf-8 -*-

import copy
import pytest

from opennmt_tokenizer import Tokenizer, BPELearner, SentencePieceLearner

def test_simple():
    tokenizer = Tokenizer("aggressive", joiner_annotate=True, joiner_new=True)
    text = "Hello World!"
    tokens = tokenizer.tokenize(text)
    assert tokens == [u"Hello", u"World", u"￭", u"!"]
    detok = tokenizer.detokenize(tokens)
    assert detok == text

def test_mode():
    tokenizer = Tokenizer(mode="char")
    assert tokenizer.tokenize("hello") == [u"h", u"e", u"l", u"l", u"o"]

def test_custom_joiner():
    tokenizer = Tokenizer("aggressive", joiner="•", joiner_annotate=True)
    tokens = tokenizer.tokenize("Hello World!")
    assert tokens == [u"Hello", u"World", u"•!"]

def test_segment_alphabet():
    tokenizer = Tokenizer(mode="aggressive", segment_alphabet=["Han"])
    tokens = tokenizer.tokenize("測試 abc")
    assert tokens == [u"測", u"試", u"abc"]

def test_named_arguments():
    tokenizer = Tokenizer(mode="aggressive", joiner_annotate=True)
    text = "Hello World!"
    tokens = tokenizer.tokenize(text=text)
    assert tokens == [u"Hello", u"World", u"￭!"]
    assert text == tokenizer.detokenize(tokens=tokens)

def test_deepcopy():
    text = "Hello World!"
    tok1 = Tokenizer("aggressive")
    tokens1 = tok1.tokenize(text)
    tok2 = copy.deepcopy(tok1)
    tokens2 = tok2.tokenize(text)
    assert tokens1 == tokens2
    del tok1
    tokens2 = tok2.tokenize(text)
    assert tokens1 == tokens2

def test_detok_with_ranges():
    tokenizer = Tokenizer("conservative")
    text, ranges = tokenizer.detokenize_with_ranges(["a", "b"])
    assert text == "a b"
    assert len(ranges) == 2
    assert ranges[0] == (0, 0)
    assert ranges[1] == (2, 2)

def test_bpe_learner(tmpdir):
    tokenizer = Tokenizer("aggressive", joiner_annotate=True)
    learner = BPELearner(tokenizer=tokenizer, symbols=2, min_frequency=1)
    learner.ingest("hello world")
    model_path = str(tmpdir.join("bpe.model"))
    tokenizer = learner.learn(model_path)
    with open(model_path) as model:
        assert model.read() == "#version: 0.2\ne l\nel l\n"
    tokens = tokenizer.tokenize("hello")
    assert tokens == [u"h￭", u"ell￭", u"o"]

def test_sp_learner(tmpdir):
    learner = SentencePieceLearner(vocab_size=17, character_coverage=0.98)
    learner.ingest("hello word! how are you?")
    model_path = str(tmpdir.join("sp.model"))
    tokenizer = learner.learn(model_path)
    tokens = tokenizer.tokenize("hello")
    assert tokens == [u"▁h", u"e", u"l", u"l", u"o"]
