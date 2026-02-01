CREATE TABLE users (
    "id" UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" UUID PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" JSONB,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" UUID PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" JSONB,
    "tags" TEXT[],
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" UUID PRIMARY KEY,
    "forId" UUID NOT NULL,
    "threadId" UUID NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "chatType" TEXT,
    "query" TEXT,
    "response" TEXT,
    "metadata" JSONB,
    "traceId" TEXT,
    "createdAt" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    "id" UUID PRIMARY KEY,
    "kbName" TEXT,
    "kbInfo" TEXT,
    "vsType" TEXT,
    "embedModel" TEXT,
    "fileCount" INT DEFAULT 0,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_file (
    "id" UUID PRIMARY KEY,
    "fileName" TEXT,
    "fileExt" TEXT,
    "kbName" TEXT,
    "documentLoaderName" TEXT,
    "textSplitterName" TEXT,
    "fileVersion" INT DEFAULT 1,
    "fileMtime" FLOAT DEFAULT 0.0,
    "fileSize" INT DEFAULT 0,
    "customDocs" BOOLEAN DEFAULT FALSE,
    "docsCount" INT DEFAULT 0,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS file_doc (
    "id" UUID PRIMARY KEY,
    "kbName" TEXT,
    "fileName" TEXT,
    "docId" TEXT,
    "metadata" JSONB
);

CREATE TABLE IF NOT EXISTS user_memory (
    "id" UUID PRIMARY KEY,
    "userId" UUID,
    "memoryText" TEXT,
    "importance" INT DEFAULT 1,
    "lastUsedTime" TEXT,
    "createdAt" TEXT,
    "metadata" JSONB,
    "threadId" UUID,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE SET NULL
);