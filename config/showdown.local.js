/**
 * Local-only Pokémon Showdown configuration for ShowdownMind.
 *
 * `--no-security` disables authentication and rate limiting, so this server
 * must never listen on a public interface.
 */
exports.port = 8765;
exports.bindaddress = '127.0.0.1';
exports.workers = 1;
exports.crashguard = true;
