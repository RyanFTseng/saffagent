"use client";

import { Configuration, DefaultApi } from "@/api-client";

export function getApiUrl(){
    return "https://sn3lzs66psvmk3kdixg6djfrne0ghuau.lambda-url.us-west-1.on.aws/";
}

export default function createApiClient(){
    const apiConfig = new Configuration({
        basePath: getApiUrl(),
    })
    const api = new DefaultApi(apiConfig);
    return api;
}