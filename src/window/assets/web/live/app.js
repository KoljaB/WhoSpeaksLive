import {createLiveAppContext} from "./live_context.js";
import {installMediaCapture} from "./media_capture.js";
import {installSessionTransport} from "./session_transport.js";
import {installSavedReports} from "./saved_reports.js";
import {installMeetingChat} from "./meeting_chat.js";
import {installTranscriptTranslation} from "./transcript_translation.js";
import {installTranscriptReview} from "./transcript_review.js";
import {installSpeakerPanel} from "./speaker_panel.js";
import {installTranscriptRender} from "./transcript_render.js";
import {installLiveBindings} from "./live_bindings.js";

export function bootstrapLiveApp() {
  const context = createLiveAppContext();
  let disposed = false;
  installMediaCapture(context);
  installSessionTransport(context);
  installSavedReports(context);
  installMeetingChat(context);
  installTranscriptTranslation(context);
  installTranscriptReview(context);
  installSpeakerPanel(context);
  installTranscriptRender(context);
  installLiveBindings(context);
  for (const activate of context.activators) activate();
  import("./help_system.js")
    .then(({installHelpSystem}) => {
      if (!disposed) installHelpSystem(context);
    })
    .catch(error => {
      console.warn("Context help could not be loaded; the core live app remains available.", error);
    });
  return () => {
    disposed = true;
    context.appResources.dispose();
  };
}

const disposeLiveApp = bootstrapLiveApp();
export default disposeLiveApp;
