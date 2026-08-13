import { NextResponse } from 'next/server';
import { DatabaseSync } from 'node:sqlite';
import path from 'node:path';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  let db: DatabaseSync | undefined;

  try {
    const databasePath =
      process.env.CAREERPATH_DB_PATH ||
      path.resolve(process.cwd(), '..', 'backend', 'careerpath.db');

    db = new DatabaseSync(databasePath);

    const row = db
      .prepare(`
        SELECT
          COUNT(*) AS total_calls,
          SUM(
            CASE
              WHEN outcome = 'SUCCESS' THEN 1
              ELSE 0
            END
          ) AS successful_calls,
          SUM(
            CASE
              WHEN outcome = 'FAILED' THEN 1
              ELSE 0
            END
          ) AS failed_calls
        FROM call_outcomes
      `)
      .get() as {
        total_calls: number;
        successful_calls: number | null;
        failed_calls: number | null;
      };

    return NextResponse.json(
      {
        totalCalls: Number(row.total_calls || 0),
        successfulCalls: Number(row.successful_calls || 0),
        failedCalls: Number(row.failed_calls || 0),
      },
      {
        headers: {
          'Cache-Control': 'no-store, no-cache, must-revalidate',
        },
      }
    );
  } catch (error) {
    console.error('Failed to load call statistics:', error);

    return NextResponse.json(
      {
        error: 'Unable to load call statistics',
      },
      {
        status: 500,
        headers: {
          'Cache-Control': 'no-store',
        },
      }
    );
  } finally {
    db?.close();
  }
}