CREATE TABLE IF NOT EXISTS holdings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    shares      REAL    NOT NULL CHECK (shares > 0),
    cost_basis  REAL    NOT NULL CHECK (cost_basis >= 0),
    purchased_at DATE   NOT NULL,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    holding_id  INTEGER REFERENCES holdings(id),
    action      TEXT    NOT NULL CHECK (action IN ('BUY', 'SELL')),
    shares      REAL    NOT NULL CHECK (shares > 0),
    price       REAL    NOT NULL CHECK (price >= 0),
    executed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes       TEXT
);
