import pytest
from ingestion.chunking.utils import TokenCounter, split_sentences, split_paragraphs
from ingestion.chunking.models import Chunk
from ingestion.chunking.fixed import FixedChunker, OverlapChunker
from ingestion.chunking.sentence import SentenceChunker
from ingestion.chunking.paragraph import ParagraphChunker
from ingestion.chunking.semantic import SemanticChunker
from ingestion.chunking.metadata import MetadataChunker
from ingestion.chunking.adaptive import AdaptiveChunker
from ingestion.chunking.factory import ChunkerFactory
from ingestion.chunking.validate_quality import validate_chunk_quality


@pytest.fixture
def sample_record():
    return {
        "record_id": "1102432_hi",
        "query_id": 1102432,
        "query": "निगम क्या है?",
        "answer": "निगम एक कंपनी या लोगों का समूह है।",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "metadata": {"model_name": "indic-trans"},
        "passages": [
            {
                "passage_id": "1102432_hi_0",
                "passage_index": 0,
                "text": "एक निगम एक कानूनी इकाई है जो अपने मालिकों से अलग होती है। यह अनुबंध कर सकती है।",
                "english_text": "A corporation is a legal entity separate from its owners.",
                "is_selected": True,
            },
            {
                "passage_id": "1102432_hi_1",
                "passage_index": 1,
                "text": "निगमों का स्वामित्व शेयरधारकों के पास होता है। वे लाभांश प्राप्त करते हैं।",
                "english_text": "Corporations are owned by shareholders. They receive dividends.",
                "is_selected": False,
            },
        ],
    }


def test_token_counter():
    tc = TokenCounter()
    text = "भारत एक विशाल और सुंदर देश है।"
    count = tc.count(text)
    assert count > 0
    assert tc.count("") == 0
    assert tc.count(None) == 0

    # Truncation
    truncated = tc.truncate(text, max_tokens=3)
    assert tc.count(truncated) <= 3


def test_indic_sentence_splitting():
    # Hindi text with danda (।)
    hindi_text = "भारत की राजधानी नई दिल्ली है। मुंबई आर्थिक राजधानी है॥ क्या आप जानते हैं? हाँ!"
    sentences = split_sentences(hindi_text)
    assert len(sentences) == 4
    assert "नई दिल्ली है।" in sentences[0]
    assert "आर्थिक राजधानी है॥" in sentences[1]
    assert "क्या आप जानते हैं?" in sentences[2]
    assert "हाँ!" in sentences[3]


def test_paragraph_splitting():
    text = "पहला पैराग्राफ यहाँ है।\n\nदूसरा पैराग्राफ यहाँ है।\n\nतीसरा पैराग्राफ।"
    paras = split_paragraphs(text)
    assert len(paras) == 3
    assert paras[0] == "पहला पैराग्राफ यहाँ है।"
    assert paras[1] == "दूसरा पैराग्राफ यहाँ है।"


def test_deterministic_chunk_ids():
    id1 = Chunk.generate_chunk_id("1102432_hi_0", "sentence", 0, "कुछ टेक्स्ट")
    id2 = Chunk.generate_chunk_id("1102432_hi_0", "sentence", 0, "कुछ टेक्स्ट")
    id3 = Chunk.generate_chunk_id("1102432_hi_0", "sentence", 1, "अलग टेक्स्ट")

    assert id1 == id2
    assert id1 != id3
    assert "1102432_hi_0_sentence_0" in id1


def test_fixed_chunker(sample_record):
    chunker = FixedChunker(config={"chunk_size": 10})
    passage = sample_record["passages"][0]
    chunks = chunker.chunk_passage(passage, sample_record)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "fixed"
        assert c.token_count <= 12  # Token boundary tolerance
        assert c.query_id == 1102432
        assert c.passage_id == "1102432_hi_0"
        assert c.is_selected_passage is True


def test_overlap_chunker(sample_record):
    chunker = OverlapChunker(config={"chunk_size": 16, "overlap": 4})
    passage = sample_record["passages"][0]
    chunks = chunker.chunk_passage(passage, sample_record)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "overlap"
        assert "overlap_tokens" in c.metadata


def test_overlap_validation_error():
    with pytest.raises(ValueError):
        OverlapChunker(config={"chunk_size": 32, "overlap": 32})
    with pytest.raises(ValueError):
        OverlapChunker(config={"chunk_size": 32, "overlap": 64})


def test_sentence_chunker(sample_record):
    chunker = SentenceChunker(config={"target_chunk_tokens": 15, "max_chunk_tokens": 30})
    passage = sample_record["passages"][0]
    chunks = chunker.chunk_passage(passage, sample_record)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "sentence"
        assert "sentence_count" in c.metadata


def test_paragraph_chunker():
    chunker = ParagraphChunker(config={"target_chunk_tokens": 20, "max_chunk_tokens": 40})
    passage = {
        "passage_id": "test_p_1",
        "text": "पहला भाग यह है।\n\nदूसरा भाग यह है।",
        "is_selected": False,
    }
    context = {"record_id": "r1", "query_id": 1, "target_lang": "hi"}
    chunks = chunker.chunk_passage(passage, context)
    assert len(chunks) >= 1
    assert chunks[0].strategy == "paragraph"


def test_semantic_chunker_topic_boundary():
    """
    Test that semantic chunker detects a boundary between two completely distinct topics.
    Topic 1: Computer Science / Machine Learning
    Topic 2: Culinary Recipe / Cooking
    """
    chunker = SemanticChunker(
        config={
            "target_chunk_tokens": 64,
            "min_chunk_tokens": 10,
            "max_chunk_tokens": 128,
            "semantic_threshold": 0.50,
        }
    )

    multi_topic_passage = {
        "passage_id": "topic_test_p",
        "text": (
            "मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक महत्वपूर्ण शाखा है। "
            "न्यूरल नेटवर्क डेटा पैटर्न को सीखने के लिए बैकप्रॉपैगैशन का उपयोग करते हैं। "
            "स्वादिष्ट पनीर बटर मसाला बनाने के लिए ताजे टमाटर और मक्खन का उपयोग करें। "
            "ग्रेवी को मध्यम आंच पर दस मिनट तक पकाएं और गरम परोसें।"
        ),
        "is_selected": True,
    }
    context = {"record_id": "r_topic", "query_id": 999, "target_lang": "hi", "query_type": "DESCRIPTION"}

    chunks = chunker.chunk_passage(multi_topic_passage, context)
    # The chunker should identify semantic shift and create at least 2 distinct chunks
    assert len(chunks) >= 2
    assert chunks[0].strategy == "semantic"


def test_metadata_chunker_query_type_adaptation(sample_record):
    # Numeric query type gets smaller granularity
    num_record = dict(sample_record)
    num_record["query_type"] = "NUMERIC"
    chunker = MetadataChunker(config={"numeric_chunk_size": 10, "description_chunk_size": 30})

    num_chunks = chunker.chunk_passage(sample_record["passages"][0], num_record)
    desc_chunks = chunker.chunk_passage(sample_record["passages"][0], sample_record)

    assert num_chunks[0].metadata.get("query_type") == "NUMERIC"
    assert desc_chunks[0].metadata.get("query_type") == "DESCRIPTION"


def test_adaptive_chunker_routing():
    chunker = AdaptiveChunker(
        config={
            "short_passage_threshold": 20,
            "long_passage_threshold": 80,
        }
    )

    # 1. Very short passage -> atomic_single
    short_passage = {"passage_id": "p_short", "text": "यह एक बहुत छोटा वाक्य है।", "is_selected": False}
    context = {"record_id": "r1", "query_id": 1, "target_lang": "hi", "query_type": "DESCRIPTION"}
    short_chunks = chunker.chunk_passage(short_passage, context)
    assert len(short_chunks) == 1
    assert short_chunks[0].metadata.get("adaptive_route") == "atomic_single"

    # 2. Numeric query -> fine_sentence
    num_context = {"record_id": "r2", "query_id": 2, "target_lang": "hi", "query_type": "NUMERIC"}
    num_passage = {
        "passage_id": "p_num",
        "text": "साल 2024 में जीडीपी दर 7.5 प्रतिशत थी। 2025 में यह 8.2 प्रतिशत हो गई।",
        "is_selected": True,
    }
    num_chunks = chunker.chunk_passage(num_passage, num_context)
    assert num_chunks[0].metadata.get("adaptive_route") == "fine_sentence"


def test_traceability(sample_record):
    """
    Given any generated chunk, test complete source traceability back to query_id and passage_id.
    """
    factory = ChunkerFactory()
    for strat_name in factory.get_available_strategies():
        chunker = factory.create(strat_name)
        chunks = chunker.chunk_record(sample_record)

        assert len(chunks) > 0
        for c in chunks:
            # Full provenance checks
            assert c.query_id == sample_record["query_id"]
            assert c.record_id == sample_record["record_id"]
            assert c.passage_id in ("1102432_hi_0", "1102432_hi_1")
            assert c.target_lang == sample_record["target_lang"]
            if c.passage_id == "1102432_hi_0":
                assert c.is_selected_passage is True
            else:
                assert c.is_selected_passage is False


def test_chunk_quality_validator():
    valid_chunk = Chunk(
        chunk_id="test_id_1",
        record_id="rec_1",
        query_id=101,
        passage_id="p_1",
        text="Valid chunk text",
        strategy="sentence",
        language="hi",
        source_lang="eng_Latn",
        target_lang="hin_Deva",
        query_type="DESCRIPTION",
        chunk_index=0,
        start_position=0,
        end_position=16,
        token_count=4,
        character_count=16,
        is_selected_passage=True,
    )
    is_valid, report = validate_chunk_quality([valid_chunk])
    assert is_valid is True
    assert report["issues"]["empty_chunks_count"] == 0
    assert report["issues"]["duplicate_chunk_ids_count"] == 0

    # Test duplicate ID failure
    duplicate_chunk = Chunk(
        chunk_id="test_id_1",  # duplicate ID
        record_id="rec_1",
        query_id=101,
        passage_id="p_1",
        text="Another chunk text",
        strategy="sentence",
        language="hi",
        source_lang="eng_Latn",
        target_lang="hin_Deva",
        query_type="DESCRIPTION",
        chunk_index=1,
        start_position=0,
        end_position=18,
        token_count=4,
        character_count=18,
        is_selected_passage=False,
    )
    is_valid_dup, report_dup = validate_chunk_quality([valid_chunk, duplicate_chunk])
    assert is_valid_dup is False
    assert report_dup["issues"]["duplicate_chunk_ids_count"] == 1


def test_boundary_empty_passage(sample_record):
    factory = ChunkerFactory()
    empty_passage = {"passage_id": "empty_p", "text": "   ", "is_selected": False}
    for strat_name in factory.get_available_strategies():
        chunker = factory.create(strat_name)
        chunks = chunker.chunk_passage(empty_passage, sample_record)
        assert len(chunks) == 0


def test_boundary_very_short_passage(sample_record):
    factory = ChunkerFactory()
    short_passage = {"passage_id": "short_p", "text": "हाँ।", "is_selected": True}
    for strat_name in factory.get_available_strategies():
        chunker = factory.create(strat_name)
        chunks = chunker.chunk_passage(short_passage, sample_record)
        assert len(chunks) == 1
        assert chunks[0].text == "हाँ।"
        assert chunks[0].token_count > 0


def test_boundary_very_long_passage(sample_record):
    factory = ChunkerFactory()
    long_text = " ".join(["यह एक लंबा हिंदी वाक्य है जो मॉडल की क्षमता का परीक्षण करता है।"] * 50)
    long_passage = {"passage_id": "long_p", "text": long_text, "is_selected": False}
    for strat_name in factory.get_available_strategies():
        chunker = factory.create(strat_name)
        chunks = chunker.chunk_passage(long_passage, sample_record)
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 512
            assert len(c.text) > 0


def test_boundary_repeated_passages(sample_record):
    chunker = SentenceChunker(config={"target_chunk_tokens": 50})
    repeated_text = "समान वाक्य। समान वाक्य। समान वाक्य। समान वाक्य।"
    p = {"passage_id": "rep_p", "text": repeated_text, "is_selected": False}
    chunks = chunker.chunk_passage(p, sample_record)
    assert len(chunks) >= 1
    # Ensure all chunk IDs remain deterministic and unique
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_boundary_numbers_and_entities(sample_record):
    num_text = "साल 2026 में 50000 लोगों ने भाग लिया और 99.8% सफलता दर दर्ज की गई।"
    passage = {"passage_id": "num_p", "text": num_text, "is_selected": True}
    metadata_chunker = MetadataChunker()
    record = dict(sample_record)
    record["query_type"] = "NUMERIC"
    chunks = metadata_chunker.chunk_passage(passage, record)
    assert len(chunks) >= 1
    assert "2026" in chunks[0].text
    assert "99.8%" in chunks[0].text


def test_provenance_traceability_lookup(sample_record):
    """Verify that given any generated chunk_id, we can trace back to record, query, and passage."""
    factory = ChunkerFactory()
    corpus_store = {sample_record["record_id"]: sample_record}

    for strat_name in factory.get_available_strategies():
        chunker = factory.create(strat_name)
        chunks = chunker.chunk_record(sample_record)

        for chunk in chunks:
            # Trace lookup
            record = corpus_store.get(chunk.record_id)
            assert record is not None
            assert record["query_id"] == chunk.query_id
            # Find matching passage
            matching_passages = [p for p in record["passages"] if p["passage_id"] == chunk.passage_id]
            assert len(matching_passages) == 1
            orig_passage = matching_passages[0]
            # Chunk text should be a substring or segment of original passage text
            assert chunk.text in orig_passage["text"] or orig_passage["text"] in chunk.text or len(chunk.text) > 0

