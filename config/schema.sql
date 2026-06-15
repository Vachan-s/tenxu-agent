-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Apparel table
CREATE TABLE IF NOT EXISTS apparel (
    style_id                    TEXT PRIMARY KEY,
    product_name                TEXT,
    website_name                TEXT,
    tagline                     TEXT,
    category                    TEXT,
    sub_category                TEXT,
    gender                      TEXT,
    activity                    TEXT,
    best_suited_for             TEXT,
    collection                  TEXT,
    available_sizes             TEXT,
    available_colours           TEXT,
    mrp                         NUMERIC,
    selling_price               NUMERIC,
    fit                         TEXT,
    sizing_note                 TEXT,
    apparel_type                TEXT,
    fabric                      TEXT,
    fabric_description          TEXT,
    pattern_texture             TEXT,
    pattern_texture_description TEXT,
    stitching_description       TEXT,
    breathability               TEXT,
    stretch                     TEXT,
    fabric_feel                 TEXT,
    reflective_elements         TEXT,
    neck_type                   TEXT,
    sleeve_type                 TEXT,
    inseam_length               TEXT,
    rise                        TEXT,
    pockets                     TEXT,
    drawcords                   TEXT,
    waistband                   TEXT,
    zip_enclosure               TEXT,
    description                 TEXT,
    wash_care                   TEXT,
    expert_review               TEXT,
    pair_with                   TEXT,
    country_of_origin           TEXT,
    embedding                   vector(384)
);

-- Footwear table
CREATE TABLE IF NOT EXISTS footwear (
    style_id            TEXT PRIMARY KEY,
    product_name        TEXT,
    website_name        TEXT,
    tagline             TEXT,
    category            TEXT,
    gender              TEXT,
    activity            TEXT,
    best_suited_for     TEXT,
    available_sizes     TEXT,
    available_colours   TEXT,
    mrp                 NUMERIC,
    selling_price       NUMERIC,
    fit                 TEXT,
    feet_type_suited    TEXT,
    lacing_closure      TEXT,
    lace_material       TEXT,
    upper_material      TEXT,
    upper_features      TEXT,
    tongue_material     TEXT,
    tongue_features     TEXT,
    heel_material       TEXT,
    heel_features       TEXT,
    midsole_material    TEXT,
    midsole_features    TEXT,
    insole_type         TEXT,
    insole_material     TEXT,
    insole_features     TEXT,
    outsole_material    TEXT,
    outsole_features    TEXT,
    forefoot_toe        TEXT,
    spikes              TEXT,
    waterproof          TEXT,
    cushioning          TEXT,
    stability           TEXT,
    breathability       TEXT,
    weight              TEXT,
    reflective_details  TEXT,
    description         TEXT,
    care_instructions   TEXT,
    expert_review       TEXT,
    country_of_origin   TEXT,
    embedding           vector(384)
);

-- Accessories table
CREATE TABLE IF NOT EXISTS accessories (
    style_id            TEXT PRIMARY KEY,
    product_name        TEXT,
    website_name        TEXT,
    category            TEXT,
    gender              TEXT,
    activity            TEXT,
    best_suited_for     TEXT,
    available_sizes     TEXT,
    available_colours   TEXT,
    mrp                 NUMERIC,
    selling_price       NUMERIC,
    material            TEXT,
    ventilation         TEXT,
    cushioning          TEXT,
    functional_features TEXT,
    description         TEXT,
    panels              TEXT,
    structure_type      TEXT,
    print_embroidery    TEXT,
    bill_visor          TEXT,
    closure_type        TEXT,
    sweatband           TEXT,
    circumference       TEXT,
    fit                 TEXT,
    care_instructions   TEXT,
    expert_review       TEXT,
    country_of_origin   TEXT,
    embedding           vector(384)
);

-- Vector search indexes
CREATE INDEX IF NOT EXISTS apparel_embedding_idx
ON apparel USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

CREATE INDEX IF NOT EXISTS footwear_embedding_idx
ON footwear USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

CREATE INDEX IF NOT EXISTS accessories_embedding_idx
ON accessories USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);