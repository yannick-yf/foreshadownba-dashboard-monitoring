import { NextResponse } from 'next/server';
import { fetchNBAGamesData } from '@/lib/s3';

export async function GET() {
  try {
    const games = await fetchNBAGamesData();
    return NextResponse.json(games);
  } catch (error) {
    console.error('Error fetching NBA games:', error);
    return NextResponse.json(
      { error: 'Failed to fetch NBA games data' },
      { status: 500 }
    );
  }
}

export const dynamic = 'force-dynamic';
export const revalidate = 300; // Revalidate every 5 minutes
