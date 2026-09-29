import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { ReactiveFormsModule } from '@angular/forms';
import { HTTP_INTERCEPTORS, HttpClientModule } from '@angular/common/http';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { OverlayModule } from '@angular/cdk/overlay';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { CardPreviewComponent } from './components/card-preview/card-preview.component';
import { CardFormComponent } from './components/card-form/card-form.component';
import { CardService } from './services/card.service';
import { AuthInterceptor } from './services/auth.interceptor';
import { LoginPageComponent } from './pages/login-page/login-page.component';
import { CreatePageComponent } from './pages/create-page/create-page.component';
import { SetBuilderPageComponent } from './pages/set-builder-page/set-builder-page.component';
import { GalleryPageComponent } from './pages/gallery-page/gallery-page.component';
import { VotePageComponent } from './pages/vote-page/vote-page.component';
import { EventHistoryPageComponent } from './pages/event-history-page/event-history-page.component';
import { AdminPageComponent } from './pages/admin-page/admin-page.component';
import { PoolPageComponent } from './pages/pool-page/pool-page.component';
import { QueueBadgeComponent } from './components/queue-badge/queue-badge.component';
import { MediaPipe } from './pipes/media.pipe';
import { TiltDirective } from './directives/tilt.directive';
import { CardSlotComponent } from './components/card-slot/card-slot.component';
import { SetRowComponent } from './components/set-row/set-row.component';
import { WinnersBannerComponent } from './components/winners-banner/winners-banner.component';
import { PoolAdminComponent } from './components/pool-admin/pool-admin.component';

@NgModule({
  declarations: [
    AppComponent,
    CardPreviewComponent,
    CardFormComponent,
    LoginPageComponent,
    CreatePageComponent,
    SetBuilderPageComponent,
    GalleryPageComponent,
    VotePageComponent,
    EventHistoryPageComponent,
    AdminPageComponent,
    PoolPageComponent,
    QueueBadgeComponent,
    CardSlotComponent,
    SetRowComponent,
    WinnersBannerComponent,
    PoolAdminComponent,
    MediaPipe,
    TiltDirective
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    ReactiveFormsModule,
    HttpClientModule,
    MatProgressSpinnerModule,
    OverlayModule
  ],
  providers: [
    CardService,
    { provide: HTTP_INTERCEPTORS, useClass: AuthInterceptor, multi: true }
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
