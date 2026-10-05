/**
 * commitlint.config.cjs — Conventional Commits.
 *
 * @see https://www.conventionalcommits.org
 */

module.exports = {
	extends: ["@commitlint/config-conventional"],
	rules: {
		"body-case": [2, "always", "sentence-case"],
		"body-full-stop": [2, "always", "."],
	},
};
