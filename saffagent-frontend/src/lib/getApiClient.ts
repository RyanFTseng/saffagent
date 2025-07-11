// lib/api-client.ts
import { Configuration, DefaultApi } from "@/api-client";

export function createApiClient() {
    const apiConfig = new Configuration({
        basePath: 'https://sn3lzs66psvmk3kdixg6djfrne0ghuau.lambda-url.us-west-1.on.aws',
    });

    return new DefaultApi(apiConfig);
}

