module.exports = {
  ci: {
    collect: {
      staticDistDir: './build',
    },
    assert: {
      preset: 'lighthouse:recommended',
      assertions: {
        'first-contentful-paint': ['error', {maxNumericValue: 1500}],
        'interactive': ['error', {maxNumericValue: 3500}],
        'categories:performance': ['error', {minScore: 0.9}],
      },
    },
  },
};
