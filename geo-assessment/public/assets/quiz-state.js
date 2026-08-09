(function (root, factory) {
  'use strict';

  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.GeoQuizState = api;
}(typeof window === 'object' ? window : null, () => {
  'use strict';

  const number = (value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.max(0, Math.floor(parsed)) : 0;
  };

  return Object.freeze({
    nextActivitySequence: (current) => number(current) + 1,
    mergeActivitySequence: (current, candidate) => Math.max(number(current), number(candidate)),
    cappedActiveDelta: (seconds) => Math.min(30, number(seconds)),
    submissionSummary: (answered, currentAnswered, selectedCount, total = 30) => {
      const completed = Math.min(number(total), Math.max(0, number(answered) - number(currentAnswered) + (number(selectedCount) > 0 ? 1 : 0)));
      return Object.freeze({ answered: completed, unanswered: Math.max(0, number(total) - completed) });
    },
    shouldCountActiveSecond: (visibilityState, hasFocus, lastActivityAt, now) => (
      visibilityState === 'visible' && Boolean(hasFocus) && number(now) - number(lastActivityAt) <= 120000
    ),
  });
}));
