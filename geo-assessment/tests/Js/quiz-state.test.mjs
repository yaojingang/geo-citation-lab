import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const state = require('../../public/assets/quiz-state.js');

test('a choice mutation owns the next sequence immediately', () => {
  assert.equal(state.nextActivitySequence('2'), 3);
});

test('an older save response cannot roll the current sequence backward', () => {
  assert.equal(state.mergeActivitySequence(5, 3), 5);
  assert.equal(state.mergeActivitySequence(3, 5), 5);
});

test('active time is sent in bounded chunks and stops after idle timeout', () => {
  assert.equal(state.cappedActiveDelta(100), 30);
  assert.equal(state.shouldCountActiveSecond('visible', true, 1_000, 120_999), true);
  assert.equal(state.shouldCountActiveSecond('visible', true, 1_000, 121_001), false);
  assert.equal(state.shouldCountActiveSecond('hidden', true, 1_000, 2_000), false);
});

test('submission summary includes the current unsaved selection without exceeding the total', () => {
  assert.deepEqual(state.submissionSummary(15, 0, 1), { answered: 16, unanswered: 14 });
  assert.deepEqual(state.submissionSummary(15, 1, 0), { answered: 14, unanswered: 16 });
  assert.deepEqual(state.submissionSummary(30, 1, 1), { answered: 30, unanswered: 0 });
});
