import {v4 as uuidv4} from "uuid";

//return session id
export function getSessionId(){
    //check if session is in browser
    if(typeof window == "undefined"){
        return "nobody";
    }

    //get session id
    let sessionId = sessionStorage.getItem("sessionId");

    //generate session id if none and store
    if(!sessionId){
        sessionId = uuidv4();
        sessionStorage.setItem("sessionId", sessionId);
    }

    return sessionId;
}

