# Data Model: 003-authentication

## Primary Tables (Better-Auth Schema)

### Table: `user`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | TEXT | PRIMARY KEY | Unique user ID |
| `name` | TEXT | NOT NULL | Full name |
| `email` | TEXT | NOT NULL, UNIQUE | User email |
| `emailVerified` | BOOLEAN | NOT NULL | Status of email verification |
| `image` | TEXT | | Profile image URL |
| `role` | TEXT | DEFAULT 'student' | User role (student, admin) |
| `background` | JSONB | | User background profile (software/hardware levels) |
| `createdAt` | TIMESTAMP | NOT NULL | |
| `updatedAt` | TIMESTAMP | NOT NULL | |

### Table: `session`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | TEXT | PRIMARY KEY | |
| `expiresAt` | TIMESTAMP | NOT NULL | |
| `token` | TEXT | NOT NULL, UNIQUE | |
| `createdAt` | TIMESTAMP | NOT NULL | |
| `updatedAt` | TIMESTAMP | NOT NULL | |
| `ipAddress` | TEXT | | |
| `userAgent` | TEXT | | |
| `userId` | TEXT | REFERENCES user(id) | |

### Table: `account`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | TEXT | PRIMARY KEY | |
| `accountId` | TEXT | NOT NULL | Internal ID from provider |
| `providerId` | TEXT | NOT NULL | 'email', 'github', etc. |
| `userId` | TEXT | REFERENCES user(id) | |
| `accessToken` | TEXT | | |
| `refreshToken` | TEXT | | |
| `idToken` | TEXT | | |
| `password` | TEXT | | Hashed password |

### Table: `verification`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | TEXT | PRIMARY KEY | |
| `identifier` | TEXT | NOT NULL | Email or phone |
| `value` | TEXT | NOT NULL | Token value |
| `expiresAt` | TIMESTAMP | NOT NULL | |
| `createdAt` | TIMESTAMP | | |
| `updatedAt` | TIMESTAMP | | |

## Relationships
- `user` (1) ↔ (M) `session`
- `user` (1) ↔ (M) `account`

## Indexes
- `idx_user_email` on `user(email)`
- `idx_session_userId` on `session(userId)`
- `idx_account_userId` on `account(userId)`
