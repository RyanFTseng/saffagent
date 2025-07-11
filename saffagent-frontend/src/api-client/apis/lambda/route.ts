import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    try {
        const response = await fetch('https://sn3lzs66psvmk3kdixg6djfrne0ghuau.lambda-url.us-west-1.on.aws/', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        return NextResponse.json({ error: 'Failed to fetch from Lambda' }, { status: 500 });
    }
}

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();

        const response = await fetch('https://sn3lzs66psvmk3kdixg6djfrne0ghuau.lambda-url.us-west-1.on.aws/agent', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        return NextResponse.json({ error: 'Failed to post to Lambda' }, { status: 500 });
    }
}