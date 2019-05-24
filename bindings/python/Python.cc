#include <fstream>
#include <memory>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <onmt/Tokenizer.h>
#include <onmt/BPE.h>
#include <onmt/SentencePiece.h>

#include <onmt/BPELearner.h>
#include <onmt/SPMLearner.h>

namespace py = pybind11;

template <typename T>
T copy(const T& v)
{
  return v;
}

template <typename T>
T deepcopy(const T& v, const py::object& dict)
{
  return v;
}

class TokenizerWrapper
{
public:
  TokenizerWrapper(const TokenizerWrapper& other)
    : _tokenizer(other._tokenizer)
  {
  }

  TokenizerWrapper(onmt::Tokenizer* tokenizer)
    : _tokenizer(tokenizer)
  {
  }

  TokenizerWrapper(const std::string& mode,
                   const std::string& bpe_model_path,
                   const std::string& sp_model_path,
                   int sp_nbest_size,
                   float sp_alpha,
                   const std::string& vocabulary_path,
                   int vocabulary_threshold,
                   const std::string& joiner,
                   bool joiner_annotate,
                   bool joiner_new,
                   bool spacer_annotate,
                   bool spacer_new,
                   bool case_markup,
                   bool no_substitution,
                   bool preserve_placeholders,
                   bool preserve_segmented_tokens,
                   bool segment_case,
                   bool segment_numbers,
                   bool segment_alphabet_change,
                   const std::vector<std::string>& segment_alphabet)
  {
    onmt::SubwordEncoder* subword_encoder = nullptr;

    if (!sp_model_path.empty())
      subword_encoder = new onmt::SentencePiece(sp_model_path, sp_nbest_size, sp_alpha);
    else if (!bpe_model_path.empty())
      subword_encoder = new onmt::BPE(bpe_model_path, joiner);

    if (subword_encoder && !vocabulary_path.empty())
      subword_encoder->load_vocabulary(vocabulary_path, vocabulary_threshold);

    int flags = 0;
    if (joiner_annotate)
      flags |= onmt::Tokenizer::Flags::JoinerAnnotate;
    if (joiner_new)
      flags |= onmt::Tokenizer::Flags::JoinerNew;
    if (spacer_annotate)
      flags |= onmt::Tokenizer::Flags::SpacerAnnotate;
    if (spacer_new)
      flags |= onmt::Tokenizer::Flags::SpacerNew;
    if (case_markup)
      flags |= onmt::Tokenizer::Flags::CaseMarkup;
    if (no_substitution)
      flags |= onmt::Tokenizer::Flags::NoSubstitution;
    if (preserve_placeholders)
      flags |= onmt::Tokenizer::Flags::PreservePlaceholders;
    if (preserve_segmented_tokens)
      flags |= onmt::Tokenizer::Flags::PreserveSegmentedTokens;
    if (segment_case)
      flags |= onmt::Tokenizer::Flags::SegmentCase;
    if (segment_numbers)
      flags |= onmt::Tokenizer::Flags::SegmentNumbers;
    if (segment_alphabet_change)
      flags |= onmt::Tokenizer::Flags::SegmentAlphabetChange;

    auto tokenizer = new onmt::Tokenizer(onmt::Tokenizer::mapMode.at(mode),
                                         subword_encoder,
                                         flags,
                                         joiner);

    for (const auto& alphabet : segment_alphabet)
      tokenizer->add_alphabet_to_segment(alphabet);

    _tokenizer.reset(tokenizer);
  }

  std::vector<std::string> tokenize(const std::string& text) const
  {
    std::vector<std::string> words;
    _tokenizer->tokenize(text, words);
    return words;
  }

  std::pair<std::string, onmt::Ranges>
  detokenize_with_ranges(const std::vector<std::string>& words, bool merge_ranges) const
  {
    onmt::Ranges ranges;
    std::string text = _tokenizer->detokenize(words, ranges, merge_ranges);
    return std::make_pair(text, ranges);
  }

  std::string detokenize(const std::vector<std::string>& words) const
  {
    return _tokenizer->detokenize(words);
  }

  const std::shared_ptr<const onmt::Tokenizer> get() const
  {
    return _tokenizer;
  }

private:
  std::shared_ptr<const onmt::Tokenizer> _tokenizer;
};

class SubwordLearnerWrapper
{
public:
  SubwordLearnerWrapper(const TokenizerWrapper* tokenizer, onmt::SubwordLearner* learner)
    : _learner(learner)
  {
    if (tokenizer)
      _tokenizer = tokenizer->get();
  }

  virtual ~SubwordLearnerWrapper() = default;

  void ingest_file(const std::string& path)
  {
    std::ifstream in(path);
    _learner->ingest(in, _tokenizer.get());
  }

  void ingest(const std::string& text)
  {
    std::istringstream in(text);
    _learner->ingest(in, _tokenizer.get());
  }

  TokenizerWrapper learn(const std::string& model_path, bool verbose)
  {
    {
      std::ofstream out(model_path);
      _learner->learn(out, nullptr, verbose);
    }

    auto new_tokenizer = create_tokenizer(model_path, _tokenizer.get());
    return TokenizerWrapper(new_tokenizer);
  }

protected:
  std::shared_ptr<const onmt::Tokenizer> _tokenizer;
  std::unique_ptr<onmt::SubwordLearner> _learner;

  // Create a new tokenizer with subword encoding configured.
  virtual onmt::Tokenizer* create_tokenizer(const std::string& model_path,
                                            const onmt::Tokenizer* tokenizer) const = 0;
};

class BPELearnerWrapper : public SubwordLearnerWrapper
{
public:
  BPELearnerWrapper(const TokenizerWrapper* tokenizer,
                    int symbols,
                    int min_frequency,
                    bool total_symbols,
                    const std::string& dict_path)
    : SubwordLearnerWrapper(tokenizer,
                            new onmt::BPELearner(false,
                                                 symbols,
                                                 min_frequency,
                                                 !dict_path.empty(),
                                                 total_symbols))
  {
    if (!dict_path.empty())
      ingest_file(dict_path);
  }

protected:
  onmt::Tokenizer* create_tokenizer(const std::string& model_path,
                                    const onmt::Tokenizer* tokenizer) const
  {
    onmt::Tokenizer* new_tokenizer = nullptr;
    if (!tokenizer)
      new_tokenizer = new onmt::Tokenizer(onmt::Tokenizer::Mode::Space);
    else
      new_tokenizer = new onmt::Tokenizer(*tokenizer);
    new_tokenizer->set_bpe_model(model_path);
    return new_tokenizer;
  }
};

static std::unordered_map<std::string, std::string> parse_kwargs(py::kwargs kwargs)
{
  std::unordered_map<std::string, std::string> map;
  map.reserve(kwargs.size());
  for (auto& item : kwargs)
    map.emplace(item.first.cast<std::string>(), py::str(item.second).cast<std::string>());
  return map;
}

static std::string create_temp_dir()
{
  py::object tempfile = py::module::import("tempfile");
  py::object mkdtemp = tempfile.attr("mkdtemp");
  return mkdtemp().cast<std::string>();
}

class SentencePieceLearnerWrapper : public SubwordLearnerWrapper
{
public:
  SentencePieceLearnerWrapper(const TokenizerWrapper* tokenizer,
                              py::kwargs kwargs)
    : SubwordLearnerWrapper(tokenizer, new onmt::SPMLearner(false, parse_kwargs(kwargs), ""))
    , _tmp_dir(create_temp_dir())
  {
    dynamic_cast<onmt::SPMLearner*>(_learner.get())->set_input_filename(_tmp_dir + "/input.txt");
  }

  ~SentencePieceLearnerWrapper()
  {
    py::object os = py::module::import("os");
    py::object rmdir = os.attr("rmdir");
    rmdir(_tmp_dir);
  }

protected:
  onmt::Tokenizer* create_tokenizer(const std::string& model_path,
                                    const onmt::Tokenizer* tokenizer) const
  {
    if (!tokenizer)
      return new onmt::Tokenizer(model_path);

    auto new_tokenizer = new onmt::Tokenizer(*tokenizer);
    new_tokenizer->set_sp_model(model_path);
    return new_tokenizer;
  }

private:
  std::string _tmp_dir;
};

PYBIND11_MODULE(opennmt_tokenizer, m)
{
  py::class_<TokenizerWrapper>(m, "Tokenizer")
    .def(py::init<std::string, std::string, std::string, int, float, std::string, int, std::string, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, std::vector<std::string>>(),
         py::arg("mode"),
         py::arg("bpe_model_path")="",
         py::arg("sp_model_path")="",
         py::arg("sp_nbest_size")=0,
         py::arg("sp_alpha")=0.1,
         py::arg("vocabulary_path")="",
         py::arg("vocabulary_threshold")=0,
         py::arg("joiner")=onmt::Tokenizer::joiner_marker,
         py::arg("joiner_annotate")=false,
         py::arg("joiner_new")=false,
         py::arg("spacer_annotate")=false,
         py::arg("spacer_new")=false,
         py::arg("case_markup")=false,
         py::arg("no_substitution")=false,
         py::arg("preserve_placeholders")=false,
         py::arg("preserve_segmented_tokens")=false,
         py::arg("segment_case")=false,
         py::arg("segment_numbers")=false,
         py::arg("segment_alphabet_change")=false,
         py::arg("segment_alphabet")=py::list())
    .def("tokenize", &TokenizerWrapper::tokenize, py::arg("text"))
    .def("detokenize", &TokenizerWrapper::detokenize, py::arg("tokens"))
    .def("detokenize_with_ranges", &TokenizerWrapper::detokenize_with_ranges,
         py::arg("tokens"), py::arg("merge_ranges")=false)
    .def("__copy__", copy<TokenizerWrapper>)
    .def("__deepcopy__", deepcopy<TokenizerWrapper>)
    ;

  py::class_<BPELearnerWrapper>(m, "BPELearner")
    .def(py::init<const TokenizerWrapper*, int, int, bool, std::string>(),
         py::arg("tokenizer")=py::none(),
         py::arg("symbols")=10000,
         py::arg("min_frequency")=2,
         py::arg("total_symbols")=false,
         py::arg("dict_path")="")
    .def("ingest", &BPELearnerWrapper::ingest, py::arg("text"))
    .def("ingest_file", &BPELearnerWrapper::ingest_file, py::arg("path"))
    .def("learn", &BPELearnerWrapper::learn,
         py::arg("model_path"), py::arg("verbose")=false)
    ;

  py::class_<SentencePieceLearnerWrapper>(m, "SentencePieceLearner")
    .def(py::init<const TokenizerWrapper*, py::kwargs>(),
         py::arg("tokenizer")=py::none())
    .def("ingest", &SentencePieceLearnerWrapper::ingest, py::arg("text"))
    .def("ingest_file", &SentencePieceLearnerWrapper::ingest_file, py::arg("path"))
    .def("learn", &SentencePieceLearnerWrapper::learn,
         py::arg("model_path"), py::arg("verbose")=false)
    ;
}
