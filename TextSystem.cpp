#pragma once
#include <array>
#include <string>
#include <vector>

struct QAPair {
  std::string prompt;
  std::array<std::string, 3> answers;
};

class QASession {
public:
  QASession() : m_currentIndex(0) {
    m_questions = {
        {"Wake up", {"...", "Why are you back?", "The answer is still no."}},
        {"I want you to accept the deal. You're in pain.",
         {"It'll get better.", "I deserve it.", "and so what"}},
        {"Let me tell you the terms again.",
         {"No.", "Leave.", "Just stop it."}},
        {"Do you want to talk instead?",
         {"I just wanna die.", "I can't speak anymore.", "..."}},
        {"Do you want this problem solved?",
         {"Yes. But I'm afraid.",
          "Yes. But I still believe it will get better.",
          "Yes. but I can't let go"}},
        {"That's natural. Accept the deal and you won't feel this way.",
         {"I want to see her again.", "I can fight back against depression",
          "I can still make a life for myself."}},
        {"All your problems will be solved.  "
         "experiencing that change.",
         {"I want you to do it.",
          "Take the part of me that broken, and lock it up.", "Heal me."}},
        {"You wont be the same. I'll need your final agreement",
         {"Do it, make me a better person.", "I agree, I have to move forward.",
          "Make me my old self again."}},
        {"...", {"...", "...", "..."}},
        {"Where am i?",
         {"You're in hell.", "You're in heaven.",
          "This was part of the deal."}},
        {"I thought only the depressed part of me was going to die?",
         {"You're not actually dead.", "You are the depressed part of you.",
          "This was also part of the deal."}},

        // Add more questions here:
        // { "Your prompt?", { "Answer A", "Answer B", "Answer C" } },
    };
  }

  // Add a question with exactly 3 answers
  void addQuestion(const std::string &prompt,
                   const std::array<std::string, 3> &answers) {
    m_questions.push_back({prompt, answers});
  }

  // --- State queries ---

  const std::string &getCurrentPrompt() const {
    return m_questions[m_currentIndex].prompt;
  }

  const std::array<std::string, 3> &getCurrentAnswers() const {
    return m_questions[m_currentIndex].answers;
  }

  const std::string &getAnswer(int index) const {
    return m_questions[m_currentIndex].answers[index];
  }

  int getCurrentIndex() const { return m_currentIndex; }

  int getTotalQuestions() const { return static_cast<int>(m_questions.size()); }

  bool isFinished() const {
    return m_currentIndex >= static_cast<int>(m_questions.size());
  }

  // --- State transitions ---

  bool advance() {
    if (m_currentIndex < static_cast<int>(m_questions.size()) - 1) {
      ++m_currentIndex;
      return true;
    }
    return false; // already at last question
  }

  bool retreat() {
    if (m_currentIndex > 0) {
      --m_currentIndex;
      return true;
    }
    return false; // already at first
  }

  void reset() { m_currentIndex = 0; }

  void jumpTo(int index) {
    if (index >= 0 && index < static_cast<int>(m_questions.size())) {
      m_currentIndex = index;
    }
  }

private:
  std::vector<QAPair> m_questions;
  int m_currentIndex;
};
